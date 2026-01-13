"""
https://ai.contentfabric.io/tagstore/iq__8JopRWbGMrQhYBgvVdgFE8xzKVz/tags?authorization=atxsjc9ZoxhGQwutApMtcfU8u26LRjUuYCUywQMeWkR7YZRN3Sp3sCTzptCnS57vDqfxS7BNTf6CH3CbZCpnQ7K7BZeE62NB5aCLBDqLz4Tkym49YEy14pACc3Zio1xNk7jCcLkfMNNy9h32Gt1Lw1FwBfKVJhyoFCrbAAHm3d7Ki8FhPg4U5iHjDKvyeDW71weT2xXi4vNWrUBVkDagebFSdWKpihMChUVwUptnc2k3VvhqBTkRs1NfRe33A2BBLV6k44G66RNTrAuFgaj9Qi%20&limit=10000&track=breakdance
"""

import requests
import argparse
from collections import defaultdict
from sklearn.metrics import balanced_accuracy_score, classification_report

annotations_url = "https://ai.contentfabric.io/tagstore/iq__8JopRWbGMrQhYBgvVdgFE8xzKVz/tags?authorization=atxsjc9ZoxhGQwutApMtcfU8u26LRjUuYCUywQMeWkR7YZRN3Sp3sCTzptCnS57vDqfxS7BNTf6CH3CbZCpnQ7K7BZeE62NB5aCLBDqLz4Tkym49YEy14pACc3Zio1xNk7jCcLkfMNNy9h32Gt1Lw1FwBfKVJhyoFCrbAAHm3d7Ki8FhPg4U5iHjDKvyeDW71weT2xXi4vNWrUBVkDagebFSdWKpihMChUVwUptnc2k3VvhqBTkRs1NfRe33A2BBLV6k44G66RNTrAuFgaj9Qi%20&limit=10000&track=breakdance"

predictions_url = "https://ai.contentfabric.io/tagstore/iq__DSuKTGURKiMdzPV2uaTGLterdHa/tags?authorization=atxsjc9ZoxhGQwutApMtcfU8u26LRjUuYCUywQMeWkR7YZRN3Sp3sCTzptCnS57vDqfxS7BNTf6CH3CbZCpnQ7K7BZeE62NB5aCLBDqLz4Tkym49YEy14pACc3Zio1xNk7jCcLkfMNNy9h32Gt1Lw1FwBfKVJhyoFCrbAAHm3d7Ki8FhPg4U5iHjDKvyeDW71weT2xXi4vNWrUBVkDagebFSdWKpihMChUVwUptnc2k3VvhqBTkRs1NfRe33A2BBLV6k44G66RNTrAuFgaj9Qi%20&limit=10000&track=breakdance"

pred_url = "https://ai.contentfabric.io/tagstore/iq__DSuKTGURKiMdzPV2uaTGLterdHa/tags?authorization=atxsjc9ZoxhGQwutApMtcfU8u26LRjUuYCUywQMeWkR7YZRN3Sp3sCTzptCnS57vDqfxS7BNTf6CH3CbZCpnQ7K7BZeE62NB5aCLBDqLz4Tkym49YEy14pACc3Zio1xNk7jCcLkfMNNy9h32Gt1Lw1FwBfKVJhyoFCrbAAHm3d7Ki8FhPg4U5iHjDKvyeDW71weT2xXi4vNWrUBVkDagebFSdWKpihMChUVwUptnc2k3VvhqBTkRs1NfRe33A2BBLV6k44G66RNTrAuFgaj9Qi%20&limit=10000&track=breakdance"

def fetch_tags(url):
    """Fetch tags from URL"""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()['tags']

def extract_all_labels(tag_str):
    """Extract all labels (can be multiple) from tag string"""
    tag_lower = tag_str.lower()
    if 'error' in tag_lower:
        tag_lower = ','.join(tag_lower.split(',')[1:])
    labels = []
    
    if 'powermove' in tag_lower:
        labels.append('powermove')
    if 'toprock' in tag_lower:
        labels.append('toprock')
    if 'footwork' in tag_lower:
        labels.append('footwork')
    if 'none' in tag_lower:
        labels.append('None')
    
    return labels if labels else ['None']

def get_ground_truth() -> dict[tuple[int, int], list[str]]:
    annotations = fetch_tags(annotations_url)
    predictions = fetch_tags(predictions_url)
    time_range_to_annotation = {}
    time_range_to_prediction = {}
    for tag in annotations:
        time_range = (tag['start_time'], tag['end_time'])
        time_range_to_annotation[time_range] = extract_all_labels(tag['tag'])
    for tag in predictions:
        time_range = (tag['start_time'], tag['end_time'])
        time_range_to_prediction[time_range] = extract_all_labels(tag['tag'])

    time_range_to_ground_truth = {}

    start, end = 0, 5000
    while (start, end) in time_range_to_annotation or (start, end) in time_range_to_prediction:
        if (start, end) in time_range_to_annotation:
            ann = time_range_to_annotation[(start, end)]
            time_range_to_ground_truth[(start, end)] = ann
        elif (start, end) in time_range_to_prediction:
            pred = time_range_to_prediction[(start, end)]
            time_range_to_ground_truth[(start, end)] = pred
        start += 5000
        end += 5000
    print('Ended at range:', start, end)

    return time_range_to_ground_truth

def main():
    gt = get_ground_truth()
    preds = fetch_tags(pred_url)
    by_label = {}
    for tag in preds:
        time_range = (tag['start_time'], tag['end_time'])
        if time_range not in gt:
            print("No ground truth for time range:", time_range)
            continue
        gt_labels = gt[time_range]
        if tag['tag'] not in by_label:
            by_label[tag['tag']] = [0, 0]
        by_label[tag['tag']][1] += 1
        if tag['tag'] in gt_labels:
            by_label[tag['tag']][0] += 1
    print(by_label)
    # compute accuracy per label as well as overall balanced accuracy
    # overall accuracy across non 'None' labels
    num = by_label['powermove'][0] + by_label['toprock'][0] + by_label['footwork'][0]
    denom = by_label['powermove'][1] + by_label['toprock'][1] + by_label['footwork'][1]
    overall_acc = num / denom if denom > 0 else 0.0
    print('Overall accuracy (excluding None):', overall_acc)

if __name__ == "__main__":
    main()