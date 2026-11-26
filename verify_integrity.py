#!/usr/bin/env python3
"""Verify every distributed research file; no ML dependencies needed."""
import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
manifest=json.loads((r/'research_manifest.json').read_text(encoding='utf-8'))
errors=[]
for name,meta in manifest.items():
    p=r/name
    if not p.is_file():errors.append('missing '+name);continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    if p.stat().st_size!=meta['size'] or h.hexdigest()!=meta['sha256']:errors.append('changed '+name)
if errors:print('\n'.join(errors));raise SystemExit(1)
print(f'PASS: {len(manifest)} original research files verified')
