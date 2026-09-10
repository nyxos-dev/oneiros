# Running Oneiros inside NyxOS — the in-OS package (I6)

**This describes the PUBLIC step.** It touches the `nyx-os` tree and puts Oneiros
in the OS, so it is NOT done autonomously — the maintainer approves it first.
Everything below is prepared and verified privately; only the final wiring is left.

## What ships

| file | size | role |
|---|---|---|
| `src/ngen_x.c` | ~10 KB | **champion** inference (transformer, self-contained, no libm) |
| `data/xformer_8M.bin` | 2.9 MB | champion weights (transformer, 1.91 bits/byte, vocab 2048) |
| `data/vocab2048.bin` | 20 KB | BPE merges + token expansions |

(`src/ngen.c` + `data/oneiros_2048.bin` are the older, smaller MLP path (2.04 bpb),
kept as a fallback. Both compile under NyxOS with no `-lm`.)

## Why it will build with NyxOS's own `cc`

- **No libm**: `ngen.c` brings its own `expf`/`logf`/`tanhf` (NyxOS libm has
  sinf/cosf/sqrtf but not exp/log/tanh; the kernel is -mno-sse). It links with
  **no `-lm`**, and the own-math output is **byte-identical** to the libm build.
- **Every libc symbol it uses exists in NyxOS userland** (`user/libc.h`): time,
  atoi/atof, strlen, memcpy/memmove, fopen/fclose/fread/fwrite/fputc/fputs/fprintf,
  malloc/free.
- **Conservative C99 only**: fixed-size arrays (no VLAs), `union` type-punning,
  stdint types, bit ops. No GCC builtins, no intrinsics, no `-march`. TinyCC-clean.

## Integration steps (the public step, on approval)

1. Add the inference as a NyxOS userland program: `user/nyxgen.c` (from `ngen.c`)
   + a `nyxgen` command, wired into the Makefile like the other user programs.
2. Put the two data files on the NyxOS disk image (e.g. `/usr/share/oneiros/`).
   `ngen` loads them with `fopen`/`fread`.
3. Build the ISO, boot in QEMU, run e.g. `nyxgen "static void " 60` and capture a
   screendump of NyxOS generating NyxOS-flavoured code **from inside the OS**.

## Honest notes

- The 2.5 MB checkpoint rides on the disk image. Future: int8-quantise the weights
  (~625 KB) for a smaller/faster in-OS model.
- Output is NyxOS-*flavoured* code (real identifiers/idioms), **not compilable** —
  this is a genuine from-scratch-neural-net tech demo, labelled as such, not a
  code-writing tool. That framing is the whole point of the honest "(a)" release.
