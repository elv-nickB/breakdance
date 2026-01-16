"""
https://ai.contentfabric.io/tagstore/iq__8JopRWbGMrQhYBgvVdgFE8xzKVz/tags?authorization=atxsjc9ZoxhGQwutApMtcfU8u26LRjUuYCUywQMeWkR7YZRN3Sp3sCTzptCnS57vDqfxS7BNTf6CH3CbZCpnQ7K7BZeE62NB5aCLBDqLz4Tkym49YEy14pACc3Zio1xNk7jCcLkfMNNy9h32Gt1Lw1FwBfKVJhyoFCrbAAHm3d7Ki8FhPg4U5iHjDKvyeDW71weT2xXi4vNWrUBVkDagebFSdWKpihMChUVwUptnc2k3VvhqBTkRs1NfRe33A2BBLV6k44G66RNTrAuFgaj9Qi%20&limit=10000&track=breakdance
"""

from src.eval import *

annotations_url = "https://ai.contentfabric.io/tagstore/iq__8JopRWbGMrQhYBgvVdgFE8xzKVz/tags?authorization=atxsjc9ZoxhGQwutApMtcfU8u26LRjUuYCUywQMeWkR7YZRN3Sp3sCTzptCnS57vDqfxS7BNTf6CH3CbZCpnQ7K7BZeE62NB5aCLBDqLz4Tkym49YEy14pACc3Zio1xNk7jCcLkfMNNy9h32Gt1Lw1FwBfKVJhyoFCrbAAHm3d7Ki8FhPg4U5iHjDKvyeDW71weT2xXi4vNWrUBVkDagebFSdWKpihMChUVwUptnc2k3VvhqBTkRs1NfRe33A2BBLV6k44G66RNTrAuFgaj9Qi%20&limit=10000&track=breakdance"

orig_predictions_url = "https://ai.contentfabric.io/tagstore/iq__DSuKTGURKiMdzPV2uaTGLterdHa/tags?authorization=atxsjc9ZoxhGQwutApMtcfU8u26LRjUuYCUywQMeWkR7YZRN3Sp3sCTzptCnS57vDqfxS7BNTf6CH3CbZCpnQ7K7BZeE62NB5aCLBDqLz4Tkym49YEy14pACc3Zio1xNk7jCcLkfMNNy9h32Gt1Lw1FwBfKVJhyoFCrbAAHm3d7Ki8FhPg4U5iHjDKvyeDW71weT2xXi4vNWrUBVkDagebFSdWKpihMChUVwUptnc2k3VvhqBTkRs1NfRe33A2BBLV6k44G66RNTrAuFgaj9Qi%20&limit=10000&track=breakdance"

pred_url = "http://localhost:8102/iq__4CizLCNJkrKEMu8cqmYL2nht7vg7/tags?authorization=atxsjc9ZoxhGQwutApMtcfU8u26LRjUuYCUywQMeWkR7YZRN3Sp3sCTzptCnS57vDqfxS7BNTf6CH3CbZCpnQ7K7BZeE62NB5aCLBDqLz4Tkym49YEy14pACc3Zio1xNk7jCcLkfMNNy9h32Gt1Lw1FwBfKVJhyoFCrbAAHm3d7Ki8FhPg4U5iHjDKvyeDW71weT2xXi4vNWrUBVkDagebFSdWKpihMChUVwUptnc2k3VvhqBTkRs1NfRe33A2BBLV6k44G66RNTrAuFgaj9Qi%20&limit=10000&track=test_breakdance"

def main():
    by_label = get_accuracy_by_label(annotations_url, orig_predictions_url, pred_url)
    print(by_label)
    # compute accuracy per label as well as overall balanced accuracy
    # overall accuracy across non 'None' labels
    num = by_label['powermove'][0] + by_label['toprock'][0] + by_label['footwork'][0]
    denom = by_label['powermove'][1] + by_label['toprock'][1] + by_label['footwork'][1]
    overall_acc = num / denom if denom > 0 else 0.0
    print('Overall accuracy (excluding None):', overall_acc)

def ms_to_timestamp(ms):
    """Convert milliseconds to HH:MM:SS:00 format"""
    seconds = ms // 1000
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:00"

def compute_regressions():
    annotations = fetch_tags(annotations_url)
    predictions = fetch_tags(pred_url)
    original_predictions = fetch_tags(orig_predictions_url)

    annotations_by_ts = {}
    predictions_by_ts = {}
    original_predictions_by_ts = {}

    for tag in annotations:
        time_range = (tag['start_time'], tag['end_time'])
        annotations_by_ts[time_range] = tag
    
    for tag in predictions:
        time_range = (tag['start_time'], tag['end_time'])
        predictions_by_ts[time_range] = tag

    for tag in original_predictions:
        time_range = (tag['start_time'], tag['end_time'])
        original_predictions_by_ts[time_range] = tag

    improvements = []
    regressions = []
    
    for time_range in annotations_by_ts:
        if time_range not in original_predictions_by_ts or time_range not in predictions_by_ts:
            continue
            
        original = original_predictions_by_ts[time_range]['tag']
        new = predictions_by_ts[time_range]['tag']

        gt = extract_all_labels(annotations_by_ts[time_range]['tag'])
        
        if new in gt and (original not in gt):
            improvements.append((time_range, gt, original, new))
        elif original in gt and (new not in gt):
            regressions.append((time_range, gt, original, new))

    print("\n" + "="*80)
    print("REGRESSION AND IMPROVEMENT ANALYSIS")
    print("="*80)
    
    print(f"\nSummary:")
    print(f"  Improvements: {len(improvements)}")
    print(f"  Regressions:  {len(regressions)}")
    print(f"  Net change:   {len(improvements) - len(regressions):+d}")
    
    if improvements:
        print("\n" + "-"*80)
        print(f"IMPROVEMENTS ({len(improvements)} total):")
        print("-"*80)
        for time_range, gt, original, new in improvements:
            start_ts = ms_to_timestamp(time_range[0])
            end_ts = ms_to_timestamp(time_range[1])
            print(f"  Time: {start_ts} | "
                  f"GT: {gt} | "
                  f"Original: {original:10s} → New: {new:10s}")
    
    if regressions:
        print("\n" + "-"*80)
        print(f"REGRESSIONS ({len(regressions)} total):")
        print("-"*80)
        for time_range, gt, original, new in regressions:
            start_ts = ms_to_timestamp(time_range[0])
            end_ts = ms_to_timestamp(time_range[1])
            print(f"  Time: {start_ts} | "
                  f"GT: {gt} | "
                  f"Original: {original:10s} → New: {new:10s}")
    
    print("\n" + "="*80)
    
    return {
        'improved': len(improvements),
        'regressed': len(regressions),
        'improvements': improvements,
        'regressions': regressions
    }


if __name__ == "__main__":
    main()