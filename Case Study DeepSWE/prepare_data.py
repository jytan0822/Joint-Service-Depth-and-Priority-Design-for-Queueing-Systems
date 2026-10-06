"""Download the public DeepSWE v1.1 trial export and extract the records used in the paper.

The case study uses GPT-5.6 trials on DeepSWE tasks (source "deep-swe"):
  * every GPT-5.6 Luna, Terra and Sol trial that the publisher includes in its score; and
  * the four GPT-5.6 Sol trials that the publisher excludes because the verifier timed out.
    The analysis counts them as failures and keeps their recorded cost.
The paper reports Sol; Luna and Terra are kept so DeepSWE_stats.ipynb can show them too.

Each file is checked against the SHA-256 in data/manifest.json before it is written, so any
change to the published records stops this script instead of silently changing the results.

Usage, from this folder:
    python prepare_data.py                        # download the export (about 51 MB)
    python prepare_data.py --source trials.json   # use a copy you downloaded yourself
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / 'data'
MODELS = ('gpt-5-6-luna', 'gpt-5-6-sol', 'gpt-5-6-terra')


def load_rows(source, url):
    if source is None:
        print(f'Downloading {url} ...')
        with urllib.request.urlopen(url) as response:
            raw = response.read()
    else:
        raw = Path(source).read_bytes()
    return json.loads(raw.decode('utf-8-sig'))['rows']


def check_and_write(records, entry):
    content = (json.dumps(records, indent=1, ensure_ascii=False) + '\n').encode('utf-8')
    if len(records) != entry['record_count'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
        raise SystemExit(
            f"{entry['file']}: the selected records differ from those used in the paper "
            f"({len(records)} found, {entry['record_count']} expected, or the checksum differs). "
            'The published export has changed, so the results cannot be reproduced from this copy.')
    (DATA / entry['file']).write_bytes(content)
    print(f"Wrote data/{entry['file']} ({len(records)} records, checksum verified)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--source', help='path to a downloaded trials.json (default: download it)')
    args = parser.parse_args()

    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    rows = [r for r in load_rows(args.source, manifest['upstream_url']) if r['source'] == 'deep-swe']
    scored = [r for r in rows if r['model'] in MODELS and r['included_in_score'] is True]
    excluded = [r for r in rows if r['model'] == 'gpt-5-6-sol' and r['included_in_score'] is False]
    check_and_write(scored, manifest)
    check_and_write(excluded, manifest['supplemental_files'][0])


if __name__ == '__main__':
    main()
