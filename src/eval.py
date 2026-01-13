import requests

def get_accuracy_by_label(annotations_url, orig_predictions_url, pred_url) -> dict[str, tuple[int, int]]:
    """
    Compute the accuracy for each outpout label.
    
    annotations_url: tagstore URL to fetch ground truth annotations from redbull labelling
    orig_predictions_url: tagstore URL to fetch original model predictions which is used to infer the ground truth labels (I know it's messy)
    pred_url: tagstore URL to fetch new model predictions to evaluate
    """

    gt = get_ground_truth(annotations_url, orig_predictions_url)
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
    return by_label

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

def get_ground_truth(annotations_url, predictions_url) -> dict[tuple[int, int], list[str]]:
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