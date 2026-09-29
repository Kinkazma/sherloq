"""Read-only, offline C2PA validation with separate binding/signature/trust states."""
import hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
BINARY=ROOT/'native/c2pa/c2patool'
SETTINGS=dict(core=dict(allowed_network_hosts=[],allow_redirects=False),verify=dict(verify_after_reading=True,verify_trust=True,verify_timestamp_trust=True,ocsp_fetch=False,remote_manifest_fetch=False),trust=dict(anchors=[]))


def snapshot(source,dest):
    before=source.stat();sha=hashlib.sha256()
    with source.open('rb') as src,dest.open('wb') as out:
        for chunk in iter(lambda:src.read(1024*1024),b''):sha.update(chunk);out.write(chunk)
    after=source.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('Le fichier a changé pendant sa lecture.')
    return sha.hexdigest()


def prepare(request):
    filename,trust_file,folder,lifetime=request;folder=Path(folder);source=Path(filename);local=folder/('image'+source.suffix.lower());sha=snapshot(source,local)
    settings=json.loads(json.dumps(SETTINGS));metadata=dict(source=str(source),sha256=sha,offline=True,tool='c2patool 0.28.0',trust_configured=False)
    if trust_file:
        pem=Path(trust_file).read_bytes()
        if len(pem)>8*1024*1024:raise ValueError('La liste de confiance dépasse 8 Mio.')
        settings['trust']=dict(trust_anchors=pem.decode('utf-8'));metadata.update(trust_configured=True,trust_sha256=hashlib.sha256(pem).hexdigest(),trust_file=trust_file)
    config=folder/'settings.json';config.write_text(json.dumps(settings));args=[str(local),'--detailed','--settings',str(config)]
    sidecar=source.with_suffix('.c2pa')
    if sidecar.is_file():
        copy=folder/'external.c2pa';metadata['sidecar_sha256']=snapshot(sidecar,copy);args+=['--external-manifest',str(copy)]
    return args,metadata


def summarize(report,metadata):
    results=report.get('validation_results',{});active=results.get('activeManifest',{})
    success={x.get('code','') for x in active.get('success',[])};failure={x.get('code','') for x in active.get('failure',[])}
    binding={'assertion.dataHash.match','assertion.bmffHash.match','assertion.boxesHash.match'}
    binding_failed=any(c.startswith(('assertion.dataHash.','assertion.bmffHash.','assertion.boxesHash.','assertion.hashedURI.')) for c in failure)
    integrity='invalid' if binding_failed else 'valid' if success&binding else 'unknown'
    signature='invalid' if any(c.startswith('claimSignature.') for c in failure) else 'valid' if 'claimSignature.validated' in success else 'unknown'
    trust='not_configured' if not metadata['trust_configured'] else 'trusted' if 'signingCredential.trusted' in success and not any(c.startswith('signingCredential.') for c in failure) else 'untrusted' if any(c.startswith('signingCredential.') for c in failure) else 'unknown'
    return dict(metadata=metadata,manifest='present' if report.get('active_manifest') else 'unknown',integrity=integrity,signature=signature,trust=trust,failures=active.get('failure',[]),report=report)


def parse(request):
    output,error,metadata=request
    if output:
        report=json.loads(output)
        if not isinstance(report,dict):raise ValueError('Réponse C2PA invalide.')
        return summarize(report,metadata)
    text=error.strip()
    if text=='Error: No claim found':state='absent'
    elif 'must fetch remote manifests from url ' in text:state='remote_unavailable'
    else:state='read_error'
    return dict(metadata=metadata,manifest=state,integrity='unknown',signature='unknown',trust='not_configured' if not metadata.get('trust_configured') else 'unknown',error=text,report=None)


def export(request):
    path,result=request;temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=Path(path).parent,delete=False) as f:
            temporary=f.name;json.dump(result,f,ensure_ascii=False,indent=2)
        os.replace(temporary,path);return path
    finally:
        if temporary and os.path.exists(temporary):os.unlink(temporary)
