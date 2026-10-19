"""Bind the actual V5 snapshot and all43 runtime sources; no device/model init."""
import hashlib,json,os,sys
from pathlib import Path
PACKAGES=('deepseek_v4_flash_csa_tnd_precision','deepseek_v4_flash_csa_tnd_full_precision')
def source_identity():
 here=Path(__file__).resolve().parent;binding=json.loads((here/'binding.json').read_text());source=Path(os.environ['SOURCE_ROOT']).resolve()
 assert str(source)==binding['source_root'] and source.name=='source' and source.parent.name=='snapshot'
 whole_helper=source.parent.parent
 assert all(Path(p or '.').resolve()!=whole_helper for p in sys.path),'whole helper must not contaminate single-layer runner'
 custom=sys.modules.get('sitecustomize')
 if custom is not None and getattr(custom,'__file__',None):assert Path(custom.__file__).resolve().parent!=whole_helper
 raw=(source.parent/'manifest.json').read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest==binding['source_manifest_sha256']
 manifest=json.loads(raw);assert len(manifest['files'])==binding['source_file_count']
 for rel,expected in manifest['files'].items():assert hashlib.sha256((source/rel).read_bytes()).hexdigest()==expected,rel
 paths={p for name in PACKAGES for p in (source/'vllm_ascend/ops/pypto'/name).rglob('*.py')};paths.update((source/'vllm_ascend/ops/pypto/variant.py',source/'vllm_ascend/ops/dsv4_csa.py'))
 tested={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
 assert len(tested)==43 and tested==binding['tested_sources'],'43 source SHA mismatch'
 assert all(manifest['files'].get(rel)==value for rel,value in tested.items())
 return {'source_manifest_sha256':digest,'source_file_count':len(manifest['files']),'tested_sources':tested,'source_root':str(source),'whole_model_sitecustomize':False}
