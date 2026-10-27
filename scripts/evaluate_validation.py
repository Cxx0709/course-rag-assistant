"""Run frozen, synthetic end-to-end checks without a model service."""
import argparse
import hashlib
import json
from pathlib import Path

from course_rag_assistant.answer import answer
from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, default=Path('eval/validation-v4.jsonl'))
    parser.add_argument('--manifest', type=Path, default=Path('eval/validation-v4-manifest.json'))
    args = parser.parse_args()
    path = args.cases
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest['sha256']:
        raise ValueError('Frozen validation cases changed')
    results = []
    for case in map(json.loads, path.read_text(encoding='utf-8').splitlines()):
        retriever = HybridRetriever([Chunk(**p) for p in case['passages']])
        result = answer(case['question'], retriever.search(case['question']))
        refused = bool(result.abstention_reason)
        correct = refused == case['expected_abstain']
        if not case['expected_abstain']:
            correct = (correct and all(p in result.text for p in case['answer_points'])
                       and any(h.chunk.doc == case['expected_doc'] for h in result.hits))
        results.append({'id': case['id'], 'passed': correct, 'expected_abstain': case['expected_abstain'],
                        'abstained': refused, 'answer': result.text, 'reason': result.abstention_reason})
    report = {'cases_sha256': digest, 'cases': len(results), 'passed': sum(r['passed'] for r in results),
              'code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(Path('src/course_rag_assistant').rglob('*.py'))},
              'details': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Validation results saved:', args.output)


if __name__ == '__main__':
    main()
