# kcare-reboot-check

Detects a running kernel that must be rebooted because of a stale patch
redirect in kernel text. Read-only. Python 3, standard library only.

## What it detects

A fixup module built before `update-2026-01-27-1` can, when unloaded, leave a
stale 5-byte jump at offset `+5` of three kernel functions:
`unoptimize_kprobe`, `optimize_all_kprobes`,
`proc_kprobes_optimization_handler`.

The jump points into memory that has since been freed. It stayed harmless
because every release up to 2026-08-31 patched those same functions and
covered it. Releases from `update-2026-08-31-1` onward no longer patch
`unoptimize_kprobe`, so the stale jump becomes live and a subsequent patch
apply can panic or hard-freeze the machine.

**This exists only in the running kernel's memory. Nothing on disk is wrong.
A reboot clears it completely and permanently.**

## Usage

Needs root, because `/proc/kcore` and the `/proc/kallsyms` addresses are
root-only.

```
sudo python3 kcare-reboot-check.py            # normal run
sudo python3 kcare-reboot-check.py --deep     # skip the boot-date shortcut
sudo python3 kcare-reboot-check.py --json     # machine-readable
sudo python3 kcare-reboot-check.py -v         # per-function detail
```

Output is one line, plus detail on stderr when the result is not clean:

```
KCARE-REBOOT-CHECK result=NEEDS_REBOOT exit=10 host=example reason=stale-redirect action=reboot
```

## Exit codes

| Code | Result | What to do |
|------|--------|------------|
| 0 | CLEAN | Nothing. This machine is not affected. |
| 10 | NEEDS_REBOOT | **Reboot this machine.** |
| 20 | INCONCLUSIVE | Could not read enough to decide. See the reason field. |
| 30 | ERROR | Not root, or bad arguments. |

Suitable for a fleet scan: run it everywhere, collect exit codes, reboot the
`10`s and investigate the `20`s.

## Use `--deep` if your patch level is pinned

By default the check short-circuits to CLEAN when the machine booted after the
fixed builds shipped, since the stale bytes cannot survive a reboot. That
assumes no older release has been applied since boot, which is not true if you
pin a patch level or run an ePortal feed that lags. In that case run `--deep`,
which always reads kernel memory instead.

## Requirements and limits

- Root.
- `kernel.kptr_restrict` must be `0` or `1`. At `2` even root sees zeroed
  addresses; the script says so and exits `20`.
- Kernel lockdown (Secure Boot with `lockdown=integrity` or
  `confidentiality`) denies `/proc/kcore`. The script reports `20` rather
  than guessing. Such a machine cannot be screened — but the reboot that
  would let you screen it is also the fix.
- x86_64 only. arm64 is not affected and exits `0` immediately.
- Kernels built without `CONFIG_OPTPROBES` are not affected and exit `0`.
- It cannot tell you whether a machine *was* primed and has already been
  rebooted. A reboot leaves nothing to see.

## What it reads

`/proc/uptime`, `/proc/kallsyms`, `/proc/kcore`,
`/sys/kernel/security/lockdown`, `/sys/kernel/debug/kprobes/list`.

It writes nothing, loads no kernel module, changes no sysctl, runs no other
program, and makes no network connection.

## Tests

```
python3 tests.py
```

Sixteen cases over the decision logic, standard library only, no root and no
kernel access needed. They do not prove the check works against a real kernel
— that was established against a crash dump and against live memory on a
machine driven into each state. They exist so that editing the byte-parsing
cannot silently break a branch, which matters because a broken branch does not
raise: it returns CLEAN.

## Reporting a result

If you get `10` or `20`, send the full output of `-v` along with the output of
`kcarectl --info`. A jump at `+5` is expected and harmless when the release you
currently have applied patches that function, so the applied release is needed
to interpret a positive result.
