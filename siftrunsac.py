import cv2
import os

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
QUERY_DIR = "query"
TRAIN_DIR = "License Plates"
OUT_DIR   = "results"

LOWE_RATIO = 0.75
MIN_MATCH_COUNT = 4  # pour afficher "AUTHORIZED" vs "NOT AUTHORIZED"

# ─────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

results = []

# Parcours des images query
for query_filename in sorted(os.listdir(QUERY_DIR)):
    query_path = os.path.join(QUERY_DIR, query_filename)
    query_img = cv2.imread(query_path)

    if query_img is None:
        continue

    gray_query = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    kp_query, desc_query = sift.detectAndCompute(gray_query, None)

    if desc_query is None or len(kp_query) == 0:
        print(f"[SKIP] {query_filename} (no descriptors)")
        continue

    best_matches = []
    best_img = None
    best_kp_train = None
    best_filename = None

    # Parcours du dataset train
    for filename in sorted(os.listdir(TRAIN_DIR)):
        train_path = os.path.join(TRAIN_DIR, filename)
        train_img = cv2.imread(train_path)

        if train_img is None:
            continue

        gray_train = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
        kp_train, desc_train = sift.detectAndCompute(gray_train, None)

        if desc_train is None or len(kp_train) == 0:
            continue

        # KNN matching (k=2) + ratio test
        raw_matches = bf.knnMatch(desc_query, desc_train, k=2)

        good = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < LOWE_RATIO * n.distance:
                good.append(m)  # on stocke directement m (pas [m])

        # garde le meilleur train
        if len(good) > len(best_matches):
            best_matches = good
            best_img = train_img
            best_kp_train = kp_train
            best_filename = filename

    if best_img is None:
        print(f"[{query_filename}] No match found.")
        continue

    # Verdict
    verdict = "AUTHORIZED" if len(best_matches) >= MIN_MATCH_COUNT else "NOT AUTHORIZED"

    # drawMatchesKnn attend une liste de listes => [[m], [m], ...]
    best_matches_knn = [[m] for m in best_matches]

    img_match = cv2.drawMatchesKnn(
        query_img, kp_query,
        best_img, best_kp_train,
        best_matches_knn, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Ajout du texte sur l'image
    title = f"{query_filename} | {verdict} | best={best_filename} | good={len(best_matches)}"
    cv2.putText(img_match, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    results.append((query_filename, img_match))

    print(f"[{query_filename}] best match -> {best_filename} | good matches = {len(best_matches)} | {verdict}")

# Affichage + sauvegarde
for query_filename, img in results:
    win_name = f"Match: {query_filename}"
    cv2.imshow(win_name, img)

    base = os.path.splitext(query_filename)[0]
    save_name = os.path.join(OUT_DIR, f"result_{base}.jpg")
    cv2.imwrite(save_name, img)
    print(f"Saved: {save_name}")

cv2.waitKey(0)
cv2.destroyAllWindows()