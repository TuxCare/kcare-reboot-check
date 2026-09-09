#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kcare-reboot-check.py -- decide whether this kernel must be rebooted.

WHAT THIS LOOKS FOR
    A fixup module built before update-2026-01-27-1 can, when it is unloaded,
    leave a stale 5-byte "jmp rel32" at offset +5 of three kernel functions:

        unoptimize_kprobe, optimize_all_kprobes, proc_kprobes_optimization_handler

    The jump points into a kpatch blob that no longer owns that address.  Every
    release between 2026-01-27 and 2026-08-31 happened to patch those same
    functions, so its own valid redirect sat on top of the stale bytes and
    nothing went wrong.  Releases from update-2026-08-31-1 onward no longer
    patch them, so the stale jump becomes live and a subsequent patch apply
    can panic or hard-freeze the box.

    It exists ONLY in the running kernel's text.  Nothing on disk is wrong.
    A reboot clears it completely and permanently.

WHAT THIS SCRIPT DOES
    Strictly read-only.  It reads, and only reads:

        /proc/uptime                      how long the kernel has been running
        /proc/kallsyms                    addresses of the functions above
        /proc/kcore                       the running kernel's memory
        /sys/kernel/security/lockdown     to explain an unreadable /proc/kcore
        /sys/kernel/debug/kprobes/list    to rule out a real kprobe (optional)

    It writes nothing, loads no module, changes no sysctl, and does not talk to
    kcarectl or to the network.  It needs root only because /proc/kcore and the
    kallsyms addresses are root-only.

    Python 3, standard library only.  The affected kernels are all el9-family,
    where python3 is always present.

CONFIDENCE, PER BRANCH
    Every branch has been exercised against live kernel memory on a machine
    driven deliberately into each state, and against a vmcore from a
    confirmed-affected host:

      - the +5 patch site, the jmp arithmetic and the three targets;
      - reading kpatch_ctx -> the applied blob -> its kpatch_file chain;
      - matching a function by its relocated daddr and reading saddr/dlen out
        of live kernel memory;
      - the orig_code offset arithmetic (three consecutive hunks chaining
        exactly: 20871 -> 20999 -> 21436);
      - the masked case -- a host whose applied patch still redirects the
        function -- read from a populated struct kpatch_undo_entry on a live
        primed kernel, reporting STALE where a clean host reports ORIGINAL.

    That path still never assumes a struct offset: it locates the entry by two
    independently computable anchors and reports INCONCLUSIVE if they do not
    agree, rather than guessing.

    Not covered: whether a host was ever primed and has since been rebooted.
    A reboot leaves nothing to see.

EXIT CODES
    0   CLEAN         not affected -- no reboot needed for this condition
    10  NEEDS_REBOOT  affected     -- reboot this machine
    20  INCONCLUSIVE  could not read enough to decide -- see the reason field
    30  ERROR         usage or internal error

    Anything other than 0 deserves a look; 10 is the reboot list.
"""

import argparse
import calendar
import json
import os
import struct
import sys
import time

VERSION = '1.0'

# The three functions a stale fixup unload could redirect.  They were always
# redirected together, by one unload, so a partial pattern is worth reporting.
FUNCS = [
    'unoptimize_kprobe',
    'optimize_all_kprobes',
    'proc_kprobes_optimization_handler',
]

# kpatch redirects a function by writing a 5-byte jmp just past the 5-byte
# ftrace NOP that opens the function, i.e. at +5.  Verified against a vmcore
# from a confirmed-affected host.
PATCH_SITE = 5
JMP_REL32 = 0xE9

# The fix first shipped in update-2026-01-27-1, so a fixup
# module built on or after that date unloads cleanly.  A kernel that booted
# after every feed had moved past that build cannot have been primed since
# boot.  Feeds lag the build by days-to-weeks, so the default carries a 90-day
# margin.  Override with --boot-cutoff if you know your feed's real cut-over.
FIX_BUILD_DATE = '2026-01-27'
DEFAULT_MARGIN_DAYS = 90

KERNEL_MIN = 0xFFFF000000000000  # plausible 64-bit kernel virtual address

CLEAN, PRIMED, INCONCLUSIVE, ERROR = 'CLEAN', 'NEEDS_REBOOT', 'INCONCLUSIVE', 'ERROR'
EXIT = {CLEAN: 0, PRIMED: 10, INCONCLUSIVE: 20, ERROR: 30}


class Bail(Exception):
    """Stop with a verdict and a machine-readable reason."""

    def __init__(self, verdict, reason, detail=''):
        super().__init__(reason)
        self.verdict, self.reason, self.detail = verdict, reason, detail


# ---------------------------------------------------------------- primitives

def u8(b, o):
    return struct.unpack_from('<B', b, o)[0]


def u16(b, o):
    return struct.unpack_from('<H', b, o)[0]


def u32(b, o):
    return struct.unpack_from('<I', b, o)[0]


def u64(b, o):
    return struct.unpack_from('<Q', b, o)[0]


def s32(b, o):
    return struct.unpack_from('<i', b, o)[0]


def cstr(b, o, n):
    raw = b[o:o + n]
    z = raw.find(b'\0')
    if z >= 0:
        raw = raw[:z]
    return raw.decode('latin-1')


def slurp(path, limit=4096):
    try:
        f = open(path, 'rb')
        try:
            return f.read(limit).decode('latin-1')
        finally:
            f.close()
    except OSError:
        return None


# ------------------------------------------------------------------- /proc/kcore

class KCore:
    """Random read access to kernel virtual memory through /proc/kcore."""

    def __init__(self):
        try:
            self.f = open('/proc/kcore', 'rb')
        except OSError as e:
            raise Bail(INCONCLUSIVE, 'kcore-unreadable',
                       '%s (%s)' % (e.strerror or str(e), lockdown_state()))
        hdr = self.f.read(64)
        if len(hdr) < 64 or hdr[:4] != b'\x7fELF' or u8(hdr, 4) != 2:
            raise Bail(INCONCLUSIVE, 'kcore-not-elf64', 'unexpected /proc/kcore header')
        phoff, phentsize, phnum = u64(hdr, 32), u16(hdr, 54), u16(hdr, 56)
        self.f.seek(phoff)
        raw = self.f.read(phentsize * phnum)
        self.segs = []
        for i in range(phnum):
            o = i * phentsize
            if u32(raw, o) != 1:  # PT_LOAD
                continue
            self.segs.append((u64(raw, o + 16),   # p_vaddr
                              u64(raw, o + 40),   # p_memsz
                              u64(raw, o + 8),    # p_offset
                              u64(raw, o + 32)))  # p_filesz
        if not self.segs:
            raise Bail(INCONCLUSIVE, 'kcore-no-segments', 'no PT_LOAD in /proc/kcore')

    def read(self, addr, size):
        """Return `size` bytes at kernel address `addr`, or None if unmapped."""
        for vaddr, memsz, off, filesz in self.segs:
            if vaddr <= addr and addr + size <= vaddr + memsz:
                delta = addr - vaddr
                if delta + size > filesz:
                    return None
                try:
                    self.f.seek(off + delta)
                    data = self.f.read(size)
                except OSError:
                    return None
                return data if len(data) == size else None
        return None

    def ptr(self, addr):
        d = self.read(addr, 8)
        return None if d is None else u64(d, 0)

    def close(self):
        try:
            self.f.close()
        except OSError:
            pass


def lockdown_state():
    s = slurp('/sys/kernel/security/lockdown')
    if not s:
        return 'lockdown unknown'
    if '[none]' in s:
        return 'lockdown=none'
    if '[integrity]' in s:
        return 'lockdown=integrity (Secure Boot) blocks /proc/kcore'
    if '[confidentiality]' in s:
        return 'lockdown=confidentiality blocks /proc/kcore'
    return 'lockdown=' + s.strip()


# ------------------------------------------------------------------ kallsyms

def read_kallsyms(wanted):
    """Map name -> address for the names in `wanted`.  Detects kptr_restrict."""
    found, zeroed, seen = {}, 0, 0
    try:
        f = open('/proc/kallsyms', 'r')
    except OSError as e:
        raise Bail(INCONCLUSIVE, 'kallsyms-unreadable', str(e))
    try:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            name = parts[2]
            if name not in wanted:
                continue
            seen += 1
            try:
                addr = int(parts[0], 16)
            except ValueError:
                continue
            if addr == 0:
                zeroed += 1
            elif name not in found:
                found[name] = addr
    finally:
        f.close()

    if seen and not found and zeroed:
        raise Bail(INCONCLUSIVE, 'kallsyms-restricted',
                   'kernel.kptr_restrict=%s hides symbol addresses; this script will '
                   'not change it -- rerun after "sysctl -w kernel.kptr_restrict=1"'
                   % (slurp('/proc/sys/kernel/kptr_restrict') or '?').strip())
    return found


# ------------------------------------------------- kpatch blob (the applied patch)

# struct kpatch_file, from kcare_file.h (stable across every shipped abiver).
KF_MAGIC, KF_ABIVER, KF_MODNAME = 0, 10, 16
KF_KPATCH_OFFSET, KF_TOTAL_SIZE = 352, 360
INFO_SIZE = 48  # struct kpatch_info


class Blob:
    """The kpatch blob currently loaded, parsed from kernel memory.

    A blob is a chain of struct kpatch_file (one per patched object: vmlinux
    plus each patched module), each carrying an array of struct kpatch_info
    describing the functions it redirects.
    """

    def __init__(self, kc, base):
        self.base = base
        self.owned = {}          # daddr -> (file_base, saddr, dlen, orig_off)
        self.vmlinux_base = None
        self.vmlinux_orig_size = 0
        self.files = 0
        self._walk(kc, base)

    def _walk(self, kc, base):
        pos, guard = base, 0
        while guard < 8192:
            hdr = kc.read(pos, 368)
            if hdr is None or hdr[:7] != b'KPATCH1':
                break
            modname = cstr(hdr, KF_MODNAME, 64)
            koff, tsize = u32(hdr, KF_KPATCH_OFFSET), u32(hdr, KF_TOTAL_SIZE)
            if tsize == 0:
                break
            orig_off = self._walk_infos(kc, pos, koff, modname)
            if modname == 'vmlinux' and self.vmlinux_base is None:
                self.vmlinux_base, self.vmlinux_orig_size = pos, orig_off
            self.files += 1
            pos += tsize
            guard += 1

    def _walk_infos(self, kc, file_base, koff, modname):
        """Walk one file's info array; return the total original-code size.

        orig_code in the undo record is the concatenation of each patched
        function's original bytes, in exactly this array order, so the running
        offset computed here is also the offset into orig_code.
        """
        addr, orig_off = file_base + koff, 0
        while True:
            chunk = kc.read(addr, INFO_SIZE * 64)
            if chunk is None:
                chunk = kc.read(addr, INFO_SIZE)
                if chunk is None:
                    return orig_off
            for i in range(len(chunk) // INFO_SIZE):
                o = i * INFO_SIZE
                daddr, saddr = u64(chunk, o), u64(chunk, o + 8)
                dlen, slen = u32(chunk, o + 16), u32(chunk, o + 20)
                if daddr == 0 and saddr == 0 and dlen == 0 and slen == 0:
                    return orig_off  # is_end_info
                if daddr and dlen:
                    self.owned[daddr] = (file_base, saddr, dlen, orig_off)
                    orig_off += dlen
            addr += len(chunk)


def find_kpatch_ctx(kc, ctx_addr):
    """Return (blob_pointer, undo_pointer) from struct kpatch_context.

    Layout is { struct mutex *lock; struct kpatch_file *patch; void *undo_data; }
    but rather than trust that, locate `patch` by finding the word that actually
    points at a KPATCH1 magic; `undo_data` is the word right after it.  Verified
    against a real host: patch at +8, undo_data at +16.
    """
    head = kc.read(ctx_addr, 64)
    if head is None:
        raise Bail(INCONCLUSIVE, 'kpatch-ctx-unreadable',
                   'cannot read kpatch_ctx at 0x%x' % ctx_addr)
    for off in range(0, 56, 8):
        cand = u64(head, off)
        if cand < KERNEL_MIN:
            continue
        magic = kc.read(cand, 8)
        if magic is not None and magic[:7] == b'KPATCH1':
            return cand, u64(head, off + 8)
    return None, None  # no patch currently applied


def orig_byte(kc, blob, undo_ptr, orig_off, want):
    """Read the byte the applied patch will restore at daddr+want on unload.

    This is the only reading that works while the machine is patched: the live
    text just shows the current patch's own redirect, but orig_code holds the
    bytes that were underneath it.  On a primed host that saved "original" is
    itself the stale jmp.

    struct kpatch_undo_entry has grown fields over the years, so nothing here
    assumes a field offset.  The entry is located by two independently
    computable anchors: its `kpatch` field must equal the vmlinux blob pointer,
    and its `orig_size` field must equal the sum of dlen over that file's info
    array.  orig_code is the word immediately before orig_size.  If both
    anchors do not land in one entry, we report INCONCLUSIVE rather than guess.
    """
    if undo_ptr is None or undo_ptr < KERNEL_MIN:
        return None, 'no-undo-record'
    head = kc.read(undo_ptr, 16)
    if head is None:
        return None, 'undo-unreadable'
    nr_entries = u32(head, 0)
    if nr_entries == 0 or nr_entries > 4096:
        return None, 'undo-empty-or-implausible'

    # entries[] starts after { uint32_t nr_entries; size_t size; } with padding.
    window = kc.read(undo_ptr + 16, 512)
    if window is None:
        return None, 'undo-entries-unreadable'

    want_kpatch, want_size = blob.vmlinux_base, blob.vmlinux_orig_size
    have_kpatch = have_size_off = None
    for off in range(0, 512 - 8, 8):
        v = u64(window, off)
        if v == want_kpatch:
            have_kpatch = off
        elif v == want_size and want_size:
            have_size_off = off
    if have_kpatch is None or have_size_off is None or have_size_off < 8:
        return None, 'undo-layout-unrecognised'

    orig_code = u64(window, have_size_off - 8)
    if orig_code < KERNEL_MIN:
        return None, 'orig-code-not-a-kernel-pointer'
    if orig_off + 16 > want_size:
        return None, 'orig-offset-out-of-range'

    # Read a window, not one byte: /proc/kcore serves vmalloc'd ranges through
    # vread(), which quietly returns zeros for anything it cannot reach.  Saved
    # original kernel code is never all-zero, so an all-zero window means the
    # read failed, not that the bytes are clean.
    span = 16
    b = kc.read(orig_code + orig_off, span)
    if b is None:
        return None, 'orig-code-unreadable'
    if b == b'\0' * span:
        return None, 'orig-code-read-returned-zeros'
    return u8(b, want), 'ok'


# ------------------------------------------------------------------ the check

def kprobe_registered(addrs):
    """True if a real kprobe sits on one of these addresses.

    An optimized kprobe also installs a 5-byte jmp, which would look exactly
    like a stale redirect.  Cheap to rule out; absent debugfs we just say so.
    """
    s = slurp('/sys/kernel/debug/kprobes/list', 65536)
    if s is None:
        return None
    for line in s.splitlines():
        tok = line.split()
        if not tok:
            continue
        try:
            a = int(tok[0], 16)
        except ValueError:
            continue
        if a in addrs:
            return True
    return False


def classify(kc, blob, undo_ptr, sym):
    """Classify one function.  Returns (state, note)."""
    text = kc.read(sym, PATCH_SITE + 8)
    if text is None:
        return 'UNREADABLE', 'kcore returned no data for text'
    if text == b'\0' * len(text):
        return 'UNREADABLE', 'kcore returned zeros (vread gave nothing)'

    if u8(text, PATCH_SITE) != JMP_REL32:
        # No redirect in the running text.  Even if the applied patch owns this
        # function, an unapplied hunk restores nothing, so there is no stale
        # jump here now.
        return 'ORIGINAL', 'no redirect at +%d' % PATCH_SITE

    if blob is None:
        # Nothing is applied, so nothing may legitimately redirect this
        # function.  A jmp here can only be left over.
        return 'STALE', 'redirect present with no patch loaded'

    target = (sym + PATCH_SITE + 5 + s32(text, PATCH_SITE + 1)) & 0xFFFFFFFFFFFFFFFF

    if sym not in blob.owned:
        # The loaded patch does not redirect this function, so this jmp is not
        # its doing.  Confirmed against the cached blobs of the two releases
        # either side of that change: K20260829_0002 patches all three of
        # these functions, K20260902_0002 patches optimize_all_kprobes and
        # proc_kprobes_optimization_handler but no longer unoptimize_kprobe --
        # which is exactly what re-exposes the stale jump.
        return 'STALE', 'redirect to 0x%x but the applied patch does not patch this function' % target

    file_base, saddr, _dlen, orig_off = blob.owned[sym]
    if saddr == 0:
        # A stack-check-only info entry: the patch inspects this function but
        # installs no redirect of its own, so the jmp is not its doing.
        return 'STALE', 'redirect to 0x%x but the applied patch installs no redirect here' % target
    if target != saddr:
        # The patch owns the function but the running kernel jumps somewhere
        # that is not the patched body.  Measured on the confirmed-affected
        # vmcore: optimize_all_kprobes was owned by the applied patch with
        # saddr 0xffffffffc21c38b0, yet the text jumped to 0xffffffffc21a71c0 --
        # inside the same blob, in its relocation table.  A "is the target
        # inside the blob" range test calls that CLEAN; comparing against the
        # exact entry point is what catches it.
        return 'STALE', 'redirect to 0x%x, not this function\'s patch entry 0x%x' % (target, saddr)

    # Masked: the loaded patch legitimately redirects this function and its own
    # jmp sits on top of whatever was underneath.  Only orig_code knows.
    if file_base != blob.vmlinux_base:
        return 'UNKNOWN', 'owned by a non-vmlinux patch file'
    val, why = orig_byte(kc, blob, undo_ptr, orig_off, PATCH_SITE)
    if val is None:
        return 'UNKNOWN', 'masked by the applied patch; %s' % why
    if val == JMP_REL32:
        return 'STALE', 'applied patch will restore a jmp at +%d on unload' % PATCH_SITE
    return 'ORIGINAL', 'applied patch will restore 0x%02x at +%d on unload' % (val, PATCH_SITE)


# ---------------------------------------------------------------------- main

def boot_epoch():
    s = slurp('/proc/uptime')
    if not s:
        return None
    try:
        return time.time() - float(s.split()[0])
    except (ValueError, IndexError):
        return None


def run(opts):
    detail = []

    arch = os.uname()[4]
    if arch not in ('x86_64', 'amd64'):
        # The kprobes part of the patch is a no-op stub off x86_64, so the
        # write in question never happened there.
        raise Bail(CLEAN, 'arch-not-affected', 'arch=%s' % arch)

    if os.geteuid() != 0:
        raise Bail(ERROR, 'not-root', 'needs root to read /proc/kcore and kallsyms')

    boot = boot_epoch()
    if boot is not None:
        detail.append('booted=%s' % time.strftime('%Y-%m-%d', time.gmtime(boot)))
        if boot >= opts['cutoff'] and not opts['deep']:
            # Nothing since this boot could have written the stale bytes, and
            # the bytes do not survive a boot.
            raise Bail(CLEAN, 'booted-after-fixed-builds',
                       'booted %s, after the %s cut-off; stale bytes cannot survive a reboot'
                       % (time.strftime('%Y-%m-%d', time.gmtime(boot)),
                          time.strftime('%Y-%m-%d', time.gmtime(opts['cutoff']))))

    syms = read_kallsyms(set(FUNCS + ['kpatch_ctx']))
    present = [f for f in FUNCS if f in syms]
    if not present:
        raise Bail(CLEAN, 'no-optprobes',
                   'none of the three symbols exist; CONFIG_OPTPROBES is off')

    probe = kprobe_registered(set(syms[f] + PATCH_SITE for f in present))
    if probe:
        raise Bail(INCONCLUSIVE, 'kprobe-registered',
                   'a live kprobe sits on a checked address; its jmp is '
                   'indistinguishable from a stale redirect')

    kc = KCore()
    try:
        blob = undo_ptr = None
        if 'kpatch_ctx' in syms:
            blob_ptr, undo_ptr = find_kpatch_ctx(kc, syms['kpatch_ctx'])
            if blob_ptr:
                blob = Blob(kc, blob_ptr)
                detail.append('patch=0x%x files=%d' % (blob_ptr, blob.files))
            else:
                detail.append('patch=none')
        else:
            detail.append('kcare=not-loaded')

        states = []
        for name in FUNCS:
            if name not in syms:
                states.append((name, 'ABSENT', 'not in kallsyms'))
                continue
            st, note = classify(kc, blob, undo_ptr, syms[name])
            states.append((name, st, note))
    finally:
        kc.close()

    detail.extend('%s=%s (%s)' % (n, s, w) for n, s, w in states)

    stale = [n for n, s, _ in states if s == 'STALE']
    unknown = [n for n, s, _ in states if s in ('UNKNOWN', 'UNREADABLE')]
    if stale:
        return PRIMED, 'stale-redirect', detail, states
    if unknown:
        return INCONCLUSIVE, 'masked-unreadable', detail, states
    return CLEAN, 'original-bytes-intact', detail, states


USAGE = """kcare-reboot-check.py [options]   (read-only, needs root)

  --deep                skip the boot-time short-circuit and always read
                        kernel memory.  Use this if your hosts may have
                        applied a pre-%s release since booting
                        (a pinned or lagging ePortal feed).
  --boot-cutoff DATE    override the safe-boot date (YYYY-MM-DD).
  --json                emit JSON instead of the one-line summary.
  -v, --verbose         print per-function detail to stderr.
  -h, --help            this text.

Exit: 0 CLEAN  10 PRIMED (reboot)  20 INCONCLUSIVE  30 ERROR
""" % FIX_BUILD_DATE


class _Parser(argparse.ArgumentParser):
    """argparse wired to this tool's contract.  Two deliberate overrides:

    error() exits 30 (ERROR), not argparse's default 2, so a typo'd flag stays
    inside the documented exit-code set instead of inventing a fifth code that
    fleet aggregation would read as neither clean nor actionable.

    format_help() returns USAGE verbatim, because that text is written for the
    sysadmin who reads this before running it as root, not for a developer.

    Please do not "simplify" either of them away.
    """

    def format_help(self):
        return USAGE

    def error(self, message):
        sys.stderr.write('%s\n\n%s' % (message, USAGE))
        sys.exit(EXIT[ERROR])


def parse_args(argv):
    p = _Parser(add_help=False)
    p.add_argument('-h', '--help', action='help')
    p.add_argument('--deep', action='store_true')
    p.add_argument('--json', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--boot-cutoff', metavar='DATE')
    a = p.parse_args(argv[1:])

    # An explicitly given cut-off is taken literally; the built-in one carries
    # the feed-propagation margin.
    date, margin = (a.boot_cutoff, 0) if a.boot_cutoff else (FIX_BUILD_DATE, DEFAULT_MARGIN_DAYS)
    try:
        base = calendar.timegm(time.strptime(date, '%Y-%m-%d'))
    except ValueError:
        p.error('bad --boot-cutoff date: %s (expected YYYY-MM-DD)' % date)
    return {'deep': a.deep, 'json': a.json, 'verbose': a.verbose,
            'cutoff': base + margin * 86400}


def main(argv):
    opts = parse_args(argv)

    try:
        verdict, reason, detail, states = run(opts)
    except Bail as b:
        verdict, reason, detail, states = b.verdict, b.reason, ([b.detail] if b.detail else []), []
    except Exception as e:  # never claim CLEAN because we crashed
        verdict, reason = INCONCLUSIVE, 'internal-error'
        detail, states = ['%s: %s' % (e.__class__.__name__, e)], []

    host = os.uname()[1]
    action = {CLEAN: 'none', PRIMED: 'reboot', INCONCLUSIVE: 'investigate', ERROR: 'fix-invocation'}[verdict]
    fn = ','.join('%s:%s' % (n, s) for n, s, _ in states) or '-'

    try:
        if opts['json']:
            print(json.dumps({'tool': 'kcare-reboot-check', 'version': VERSION,
                              'host': host, 'result': verdict, 'exit': EXIT[verdict],
                              'reason': reason, 'action': action, 'functions': fn,
                              'detail': '; '.join(detail)}, sort_keys=True))
        else:
            print('KCARE-REBOOT-CHECK result=%s exit=%d host=%s reason=%s action=%s functions=%s'
                  % (verdict, EXIT[verdict], host, reason, action, fn))
            if opts['verbose'] or verdict != CLEAN:
                for d in detail:
                    sys.stderr.write('  %s\n' % d)
        # Flush here, inside the guard, so a closed pipe surfaces as a catchable
        # error at this point.  Without it print() only fills the buffer and the
        # BrokenPipeError lands in the interpreter's shutdown flush instead,
        # where nothing can catch it and Python overrides the exit status with
        # 120 -- turning a PRIMED host into an unrecognised code.  Measured:
        # `check | head -0` returned 120 instead of 10 before this flush.
        sys.stdout.flush()
    except OSError:
        # Nobody is reading stdout (`... | head`).  The exit code is the
        # machine-readable result and has to survive that, so point the stream
        # at /dev/null: the shutdown flush then succeeds silently and the
        # verdict below is what reaches the shell.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:
            pass

    return EXIT[verdict]


if __name__ == '__main__':
    sys.exit(main(sys.argv))
