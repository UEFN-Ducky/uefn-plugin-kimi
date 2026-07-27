# Kimi

Kimi API (Moonshot Open Platform) for Settings → LLMs. Install from Store → Gateways, then enable to show Kimi under Providers & Keys.

Desktop plugin for [UEFN-Ducky](https://github.com/UEFN-Ducky/UEFN-Ducky) (`kimi`).
Install or update from **Settings → Store** in the app — do not install from a zip by hand.

## Build

```bash
py scripts/build_zip.py
```

Writes `deploy/kimi-1.0.0.ducky-plugin.zip` (scripts/ and deploy/ are not packed).

## Secrets

Never commit tokens or keys. The app stores `kimi` locally (DPAPI), not in this package.
