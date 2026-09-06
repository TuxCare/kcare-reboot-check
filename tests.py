#!/usr/bin/env python3
"""Regression tests for kcare-reboot-check.

These do not prove the check works against a real kernel -- that was done
against a vmcore from a confirmed-affected host and against live memory on a
machine driven into each state.  They exist so that editing the byte-parsing
cannot silently break a branch.

That matters because of the failure direction: a broken branch does not raise,
it returns CLEAN.  A machine that needs rebooting would be told it is fine.

Every return path of classify() is covered, plus the two anchors orig_byte()
uses to find the undo record.  Where the numbers are taken from the reference
vmcore they are marked, so the test encodes measured ground truth rather than
an invented example.

Run:  python3 tests.py
"""

import importlib.util
import struct
import sys

spec = importlib.util.spec_from_file_location('chk', 'kcare-reboot-check.py')
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)

SITE = chk.PATCH_SITE          # 5
JMP = chk.JMP_REL32            # 0xE9

# Measured on the reference vmcore (optimize_all_kprobes on the affected host).
VM_SYM = 0xffffffff9602e390    # the kernel function
VM_SADDR = 0xffffffffc21c38b0  # its patch entry, from the applied blob
VM_TARGET = 0xffffffffc21a71c0  # where the running text actually jumped
BLOB_BASE = 0xffffffffc216f000

FTRACE_NOP = b'\x0f\x1f\x44\x00\x00'
ORIG_FIRST_BYTE = 0x48         # cmpq $aggr_pre_handler, 0x40(%rdi)


def text_with_jmp(sym, target):
    """5-byte ftrace nop, then a jmp rel32 from sym+5 to target."""
    rel = (target - (sym + SITE + 5)) & 0xFFFFFFFF
    return FTRACE_NOP + bytes([JMP]) + struct.pack('<I', rel) + b'\x41' * 3


def text_original():
    return FTRACE_NOP + bytes([ORIG_FIRST_BYTE]) + b'\x81\x7f\x40\x60\xd8\x02\x96'


class FakeKC(object):
    """Serves reads from explicit (addr, bytes) regions; None outside them."""

    def __init__(self, regions=None):
        self.regions = list(regions or [])

    def add(self, addr, data):
        self.regions.append((addr, data))
        return self

    def read(self, addr, size):
        for base, data in self.regions:
            if base <= addr and addr + size <= base + len(data):
                off = addr - base
                return data[off:off + size]
        return None


class FakeBlob(object):
    def __init__(self, owned=None, vmlinux_base=BLOB_BASE, orig_size=4096):
        self.owned = owned or {}
        self.vmlinux_base = vmlinux_base
        self.vmlinux_orig_size = orig_size


def undo_record(undo_ptr, kpatch_ptr, orig_size, orig_code_ptr, nr_entries=1):
    """A synthetic undo record laid out the way orig_byte() searches for it.

    orig_byte() makes no assumption about field offsets: it scans the first
    512 bytes of entries[] for a word equal to the blob pointer and a word
    equal to the summed dlen, and takes orig_code as the word immediately
    before the latter.  These offsets match struct kpatch_undo_entry as
    compiled from undo.h (orig_code +72, orig_size +80, kpatch +136), but the
    test would pass with any layout the scan can resolve -- which is the point.
    """
    head = struct.pack('<I', nr_entries) + b'\0' * 12
    win = bytearray(512)
    struct.pack_into('<Q', win, 72, orig_code_ptr)
    struct.pack_into('<Q', win, 80, orig_size)
    struct.pack_into('<Q', win, 136, kpatch_ptr)
    return [(undo_ptr, head), (undo_ptr + 16, bytes(win))]


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


# ------------------------------------------------------- classify(), no patch

@case
def text_unreadable_is_not_clean():
    st, note = chk.classify(FakeKC(), None, None, VM_SYM)
    assert st == 'UNREADABLE', (st, note)


@case
def zero_filled_read_is_not_clean():
    """/proc/kcore serves vmalloc via vread(), which returns zeros for what it
    cannot reach.  Treating that as clean bytes would report CLEAN on an
    unreadable host."""
    kc = FakeKC([(VM_SYM, b'\0' * (SITE + 8))])
    st, note = chk.classify(kc, None, None, VM_SYM)
    assert st == 'UNREADABLE', (st, note)


@case
def original_bytes_are_clean():
    kc = FakeKC([(VM_SYM, text_original())])
    st, note = chk.classify(kc, None, None, VM_SYM)
    assert st == 'ORIGINAL' and 'no redirect' in note, (st, note)


@case
def jmp_with_no_patch_loaded_is_stale():
    kc = FakeKC([(VM_SYM, text_with_jmp(VM_SYM, VM_TARGET))])
    st, note = chk.classify(kc, None, None, VM_SYM)
    assert st == 'STALE' and 'no patch loaded' in note, (st, note)


# ---------------------------------------------------- classify(), patch loaded

@case
def jmp_in_a_function_the_patch_does_not_own_is_stale():
    """The post-2026-08-31 case: the blob stopped carrying unoptimize_kprobe,
    so any jmp there is not its doing."""
    kc = FakeKC([(VM_SYM, text_with_jmp(VM_SYM, VM_TARGET))])
    st, note = chk.classify(kc, FakeBlob(owned={}), None, VM_SYM)
    assert st == 'STALE' and 'does not patch this function' in note, (st, note)


@case
def owned_but_with_no_redirect_of_its_own_is_stale():
    owned = {VM_SYM: (BLOB_BASE, 0, 0x80, 0)}       # saddr == 0
    kc = FakeKC([(VM_SYM, text_with_jmp(VM_SYM, VM_TARGET))])
    st, note = chk.classify(kc, FakeBlob(owned=owned), None, VM_SYM)
    assert st == 'STALE' and 'installs no redirect' in note, (st, note)


@case
def target_inside_the_blob_but_not_the_patch_entry_is_stale():
    """The measured vmcore case, and the reason a range test is wrong: the
    stale target sits INSIDE the live blob, in its relocation table.  Only
    comparing against the exact patch entry catches it."""
    assert BLOB_BASE < VM_TARGET < BLOB_BASE + 28933648, 'target must be in-blob'
    owned = {VM_SYM: (BLOB_BASE, VM_SADDR, 0x80, 0)}
    kc = FakeKC([(VM_SYM, text_with_jmp(VM_SYM, VM_TARGET))])
    st, note = chk.classify(kc, FakeBlob(owned=owned), None, VM_SYM)
    assert st == 'STALE', (st, note)
    assert ('0x%x' % VM_TARGET) in note and ('0x%x' % VM_SADDR) in note, note


@case
def jmp_target_arithmetic_is_right():
    """A wrong rel32 computation would make every masked host look stale."""
    owned = {VM_SYM: (BLOB_BASE, VM_SADDR, 0x80, 0)}
    kc = FakeKC([(VM_SYM, text_with_jmp(VM_SYM, VM_SADDR))])
    st, note = chk.classify(kc, FakeBlob(owned=owned), None, VM_SYM)
    # target == saddr, so it goes on to orig_code and cannot decide from text
    assert st == 'UNKNOWN', (st, note)


@case
def non_vmlinux_owner_is_not_guessed():
    owned = {VM_SYM: (BLOB_BASE + 0x100000, VM_SADDR, 0x80, 0)}
    kc = FakeKC([(VM_SYM, text_with_jmp(VM_SYM, VM_SADDR))])
    st, note = chk.classify(kc, FakeBlob(owned=owned), None, VM_SYM)
    assert st == 'UNKNOWN' and 'non-vmlinux' in note, (st, note)


# ------------------------------------------- classify(), masked -> orig_code

def masked_kc(orig_first_byte):
    undo_ptr, orig_code = 0xffffffffc9000000, 0xffffffffc8000000
    saved = bytes([0x0f, 0x1f, 0x44, 0x00, 0x00, orig_first_byte]) + b'\x11' * 10
    kc = FakeKC([(VM_SYM, text_with_jmp(VM_SYM, VM_SADDR)), (orig_code, saved)])
    for addr, data in undo_record(undo_ptr, BLOB_BASE, 4096, orig_code):
        kc.add(addr, data)
    return kc, undo_ptr


@case
def masked_clean_host_is_clean():
    kc, undo_ptr = masked_kc(ORIG_FIRST_BYTE)
    owned = {VM_SYM: (BLOB_BASE, VM_SADDR, 0x80, 0)}
    st, note = chk.classify(kc, FakeBlob(owned=owned), undo_ptr, VM_SYM)
    assert st == 'ORIGINAL' and '0x48' in note, (st, note)


@case
def masked_primed_host_is_stale():
    """The 1414-host state: a legitimate redirect on top, a stale jmp saved
    underneath.  Getting this one wrong tells the wrong hosts they are fine."""
    kc, undo_ptr = masked_kc(JMP)
    owned = {VM_SYM: (BLOB_BASE, VM_SADDR, 0x80, 0)}
    st, note = chk.classify(kc, FakeBlob(owned=owned), undo_ptr, VM_SYM)
    assert st == 'STALE' and 'restore a jmp' in note, (st, note)


# -------------------------------------------------- orig_byte() anchor safety

@case
def empty_undo_record_is_inconclusive():
    """The reference vmcore is exactly this: allocated but never populated,
    because the panic hit mid-apply."""
    kc, undo_ptr = masked_kc(JMP)
    kc.regions = [(a, d) for a, d in kc.regions if a != undo_ptr]
    kc.add(undo_ptr, struct.pack('<I', 0) + b'\0' * 12)
    val, why = chk.orig_byte(kc, FakeBlob(), undo_ptr, 0, SITE)
    assert val is None and why == 'undo-empty-or-implausible', (val, why)


@case
def disagreeing_anchors_are_inconclusive_not_a_guess():
    kc, undo_ptr = masked_kc(JMP)
    blob = FakeBlob(vmlinux_base=0xffffffffdeadbeef & ~0xf)   # anchor won't match
    val, why = chk.orig_byte(kc, blob, undo_ptr, 0, SITE)
    assert val is None and why == 'undo-layout-unrecognised', (val, why)


@case
def zero_filled_orig_code_is_inconclusive():
    undo_ptr, orig_code = 0xffffffffc9000000, 0xffffffffc8000000
    kc = FakeKC([(orig_code, b'\0' * 16)])
    for addr, data in undo_record(undo_ptr, BLOB_BASE, 4096, orig_code):
        kc.add(addr, data)
    val, why = chk.orig_byte(kc, FakeBlob(), undo_ptr, 0, SITE)
    assert val is None and why == 'orig-code-read-returned-zeros', (val, why)


@case
def missing_undo_pointer_is_inconclusive():
    val, why = chk.orig_byte(FakeKC(), FakeBlob(), None, 0, SITE)
    assert val is None and why == 'no-undo-record', (val, why)


# ------------------------------------------------------------- exit-code map

@case
def exit_codes_are_the_documented_contract():
    assert chk.EXIT[chk.CLEAN] == 0
    assert chk.EXIT[chk.PRIMED] == 10
    assert chk.EXIT[chk.INCONCLUSIVE] == 20
    assert chk.EXIT[chk.ERROR] == 30
    assert chk.PRIMED == 'NEEDS_REBOOT', chk.PRIMED


def main():
    failed = []
    for fn in CASES:
        try:
            fn()
            print('  ok    %s' % fn.__name__)
        except AssertionError as e:
            failed.append(fn.__name__)
            print('  FAIL  %s: %s' % (fn.__name__, e))
        except Exception as e:
            failed.append(fn.__name__)
            print('  ERROR %s: %s: %s' % (fn.__name__, e.__class__.__name__, e))
    print('\n%d passed, %d failed' % (len(CASES) - len(failed), len(failed)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
