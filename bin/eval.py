"""
https://ai.contentfabric.io/tagstore/iq__8JopRWbGMrQhYBgvVdgFE8xzKVz/tags?authorization=atxsjc9ZoxhGQwutApMtcfU8u26LRjUuYCUywQMeWkR7YZRN3Sp3sCTzptCnS57vDqfxS7BNTf6CH3CbZCpnQ7K7BZeE62NB5aCLBDqLz4Tkym49YEy14pACc3Zio1xNk7jCcLkfMNNy9h32Gt1Lw1FwBfKVJhyoFCrbAAHm3d7Ki8FhPg4U5iHjDKvyeDW71weT2xXi4vNWrUBVkDagebFSdWKpihMChUVwUptnc2k3VvhqBTkRs1NfRe33A2BBLV6k44G66RNTrAuFgaj9Qi%20&limit=10000&track=breakdance
"""

from src.eval import *

annotations_url = "https://ai.contentfabric.io/tagstore/iq__8JopRWbGMrQhYBgvVdgFE8xzKVz/tags?authorization=atxsjc9ZoxhGQwutApMtcfU8u26LRjUuYCUywQMeWkR7YZRN3Sp3sCTzptCnS57vDqfxS7BNTf6CH3CbZCpnQ7K7BZeE62NB5aCLBDqLz4Tkym49YEy14pACc3Zio1xNk7jCcLkfMNNy9h32Gt1Lw1FwBfKVJhyoFCrbAAHm3d7Ki8FhPg4U5iHjDKvyeDW71weT2xXi4vNWrUBVkDagebFSdWKpihMChUVwUptnc2k3VvhqBTkRs1NfRe33A2BBLV6k44G66RNTrAuFgaj9Qi%20&limit=10000&track=breakdance"

orig_predictions_url = "https://ai.contentfabric.io/tagstore/iq__DSuKTGURKiMdzPV2uaTGLterdHa/tags?authorization=atxsjc9ZoxhGQwutApMtcfU8u26LRjUuYCUywQMeWkR7YZRN3Sp3sCTzptCnS57vDqfxS7BNTf6CH3CbZCpnQ7K7BZeE62NB5aCLBDqLz4Tkym49YEy14pACc3Zio1xNk7jCcLkfMNNy9h32Gt1Lw1FwBfKVJhyoFCrbAAHm3d7Ki8FhPg4U5iHjDKvyeDW71weT2xXi4vNWrUBVkDagebFSdWKpihMChUVwUptnc2k3VvhqBTkRs1NfRe33A2BBLV6k44G66RNTrAuFgaj9Qi%20&limit=10000&track=breakdance"

pred_url = "https://ai.contentfabric.io/tagstore/iq__DSuKTGURKiMdzPV2uaTGLterdHa/tags?authorization=atxsjc9ZoxhGQwutApMtcfU8u26LRjUuYCUywQMeWkR7YZRN3Sp3sCTzptCnS57vDqfxS7BNTf6CH3CbZCpnQ7K7BZeE62NB5aCLBDqLz4Tkym49YEy14pACc3Zio1xNk7jCcLkfMNNy9h32Gt1Lw1FwBfKVJhyoFCrbAAHm3d7Ki8FhPg4U5iHjDKvyeDW71weT2xXi4vNWrUBVkDagebFSdWKpihMChUVwUptnc2k3VvhqBTkRs1NfRe33A2BBLV6k44G66RNTrAuFgaj9Qi%20&limit=10000&track=breakdance"

def main():
    by_label = get_accuracy_by_label(annotations_url, orig_predictions_url, pred_url)
    print(by_label)
    # compute accuracy per label as well as overall balanced accuracy
    # overall accuracy across non 'None' labels
    num = by_label['powermove'][0] + by_label['toprock'][0] + by_label['footwork'][0]
    denom = by_label['powermove'][1] + by_label['toprock'][1] + by_label['footwork'][1]
    overall_acc = num / denom if denom > 0 else 0.0
    print('Overall accuracy (excluding None):', overall_acc)

if __name__ == "__main__":
    main()