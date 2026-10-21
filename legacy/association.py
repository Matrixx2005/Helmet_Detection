
def associate_person_helmet(detections):
    """
    Lien ket nguoi voi mu bao ho.
    Uu tien mu nam gan dinh dau, khong chi nam trong
    khung Person.
    """
    persons = [
        item for item in detections
        if item["class_name"].lower() == "person"
    ]

    helmets = [
        item for item in detections
        if item["class_name"].lower() == "helmet"
    ]

    no_helmets = [
        item for item in detections
        if item["class_name"].lower() == "no_helmet"
    ]

    # Tao cac cap nguoi-mu co kha nang phu hop
    candidates = []

    for person_idx, person in enumerate(persons, start=1):
        px1, py1, px2, py2 = person["bbox"]
        pw = px2 - px1
        ph = py2 - py1

        if pw <= 0 or ph <= 0:
            continue

        # Vung dau uu tien: phia tren cua khung nguoi
        head_bottom = py1 + 0.30 * ph
        head_center_x = (px1 + px2) / 2
        expected_head_y = py1 + 0.12 * ph

        for helmet_idx, helmet in enumerate(helmets):
            hx1, hy1, hx2, hy2 = helmet["bbox"]
            hx = (hx1 + hx2) / 2
            hy = (hy1 + hy2) / 2

            # Mu phai nam trong phan tren cua khung nguoi
            in_head_region = (
                px1 <= hx <= px2
                and py1 <= hy <= head_bottom
                and hy2 <= py1 + 0.38 * ph
            )

            if not in_head_region:
                continue

            # Diem cang nho, tam mu cang gan vi tri dau uoc tinh
            dx = (hx - head_center_x) / max(pw / 2, 1)
            dy = (hy - expected_head_y) / max(0.20 * ph, 1)
            score = (dx ** 2 + dy ** 2) ** 0.5

            candidates.append({
                "person_idx": person_idx,
                "helmet_idx": helmet_idx,
                "score": score
            })

    # Moi nguoi toi da mot mu, moi mu toi da mot nguoi
    candidates.sort(key=lambda x: x["score"])

    person_to_helmet = {}
    used_helmets = set()

    for candidate in candidates:
        p_idx = candidate["person_idx"]
        h_idx = candidate["helmet_idx"]

        if p_idx in person_to_helmet or h_idx in used_helmets:
            continue

        person_to_helmet[p_idx] = h_idx
        used_helmets.add(h_idx)

    associations = []

    for person_idx, person in enumerate(persons, start=1):
        px1, py1, px2, py2 = person["bbox"]
        pw = px2 - px1
        ph = py2 - py1

        matched_idx = person_to_helmet.get(person_idx)
        matched_helmet = (
            helmets[matched_idx]
            if matched_idx is not None
            else None
        )

        # Tim bang chung no_helmet trong vung dau
        no_helmet_evidence = False

        if pw > 0 and ph > 0:
            head_bottom = py1 + 0.35 * ph

            for item in no_helmets:
                nx1, ny1, nx2, ny2 = item["bbox"]
                nx = (nx1 + nx2) / 2
                ny = (ny1 + ny2) / 2

                if (
                    px1 <= nx <= px2
                    and py1 <= ny <= head_bottom
                ):
                    no_helmet_evidence = True
                    break

        associations.append({
            "person_index": person_idx,
            "person_bbox": person["bbox"],
            "helmet_detected": matched_helmet is not None,
            "helmet_bbox": (
                matched_helmet["bbox"]
                if matched_helmet is not None
                else None
            ),
            "no_helmet_evidence": no_helmet_evidence
        })

    return associations
