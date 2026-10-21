
def infer_safety_status(detections, associations):
    """
    Suy luan trang thai mu bao ho.
    Khong ket luan thieu mu neu chua co bang chung ro rang.
    """
    persons = [
        item for item in detections
        if item["class_name"].lower() == "person"
    ]

    if not persons:
        return [{
            "person_index": None,
            "status": "NO_PERSON_DETECTED",
            "reason": "Khong phat hien nguoi trong anh.",
            "rule_id": "R0"
        }]

    if not associations:
        return [{
            "person_index": None,
            "status": "UNKNOWN",
            "reason": "Chua co ket qua lien ket nguoi-mu.",
            "rule_id": "R3"
        }]

    results = []

    for association in associations:
        person_index = association["person_index"]

        if association["helmet_detected"]:
            status = "HELMET_DETECTED"
            reason = (
                "Phat hien mu trong vung dau uoc tinh; "
                "chua khang dinh chac chan mu dang duoc doi."
            )
            rule_id = "R1"

        elif association.get("no_helmet_evidence", False):
            status = "SUSPECTED_NO_HELMET"
            reason = (
                "Mo hinh phat hien bang chung no_helmet "
                "trong vung dau; can xac minh ket qua."
            )
            rule_id = "R2"

        else:
            status = "UNKNOWN"
            reason = (
                "Khong tim thay mu phu hop va chua co "
                "bang chung du ro rang de suy luan."
            )
            rule_id = "R3"

        results.append({
            "person_index": person_index,
            "status": status,
            "reason": reason,
            "rule_id": rule_id
        })

    return results
