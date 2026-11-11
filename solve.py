"""Generate the HW3 figures and numeric results. Run from any directory."""
import json
from pathlib import Path

import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import rq

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
cv.setNumThreads(1)
cv.setRNGSeed(0)
cv.ocl.setUseOpenCL(False)


def read(name, flags=cv.IMREAD_COLOR):
    image = cv.imread(str(INPUT / name), flags)
    if image is None:
        raise FileNotFoundError(name)
    return image


def rgb(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB) if image.ndim == 3 else image


def panels(filename, items, columns=2):
    rows = (len(items) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows),
                             squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, image) in zip(axes.flat, items):
        ax.imshow(rgb(image), cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=11)
    fig.savefig(OUT / filename, dpi=150, facecolor="white")
    plt.close(fig)


def warp_gray(image, alpha, swirl_degrees=0, inverse=False):
    """Forward scatter or inverse bilinear sampling for the radial/swirl map."""
    h, w = image.shape
    cx, cy = w / 2, h / 2
    y, x = np.indices(image.shape, dtype=float)
    r = np.hypot(x - cx, y - cy)
    theta = np.arctan2(y - cy, x - cx)
    max_r = np.hypot(cx, cy)
    swirl = np.deg2rad(swirl_degrees)

    if inverse:
        source_r = r**2 / alpha
        source_theta = theta - swirl * (1 - source_r / max_r)
        sx = cx + source_r * np.cos(source_theta)
        sy = cy + source_r * np.sin(source_theta)
        valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
        # Clip before indexing. The valid mask specifies the background.
        sx, sy = np.clip(sx, 0, w - 1), np.clip(sy, 0, h - 1)
        x0, y0 = sx.astype(int), sy.astype(int)
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        a, b = sx - x0, sy - y0
        value = ((1 - a) * (1 - b) * image[y0, x0]
                 + a * (1 - b) * image[y0, x1]
                 + (1 - a) * b * image[y1, x0] + a * b * image[y1, x1])
        return np.where(valid, np.rint(value), 0).astype(np.uint8), valid

    new_r = np.sqrt(alpha * r)
    new_theta = theta + swirl * (1 - r / max_r)
    xn = np.rint(cx + new_r * np.cos(new_theta)).astype(int)
    yn = np.rint(cy + new_r * np.sin(new_theta)).astype(int)
    inside = (xn >= 0) & (xn < w) & (yn >= 0) & (yn < h)
    index = yn[inside] * w + xn[inside]
    sums = np.bincount(index, weights=image[inside], minlength=h * w)
    counts = np.bincount(index, minlength=h * w)
    output = np.rint(sums / np.maximum(counts, 1)).astype(np.uint8)
    return output.reshape(h, w), (counts > 0).reshape(h, w)


def problem1():
    im = read("einstein.jpg", cv.IMREAD_GRAYSCALE)
    sweep = []
    for alpha in [200, 300, 400, 500, 600]:
        result, valid = warp_gray(im, alpha, inverse=True)
        sweep.append((f"alpha = {alpha}", result))
    panels("p1_alpha.png", [("Original", im)] + sweep, 3)
    alpha = 600
    items, gaps = [], []
    for label, angle in [("Radial", 0), ("Radial + 35 degree swirl", 35)]:
        scattered, occupied = warp_gray(im, alpha, angle)
        sampled, valid = warp_gray(im, alpha, angle, inverse=True)
        items += [(f"{label} / forward", scattered),
                  (f"{label} / bilinear", sampled)]
        gaps.append({"angle": angle, "rounding_gaps": int((valid & ~occupied).sum()),
                     "outside_source": int((~valid).sum())})
    panels("p1_results.png", items)
    return {"alpha": alpha, "gaps": gaps}


def polygon_mask(image, vertices):
    mask = np.zeros(image.shape[:2], np.uint8)
    cv.fillPoly(mask, [np.asarray(vertices, np.int32)], 255)
    return mask


def matches(a, b, masks=(None, None), ratio=0.7):
    detector = cv.SIFT_create(nfeatures=0, nOctaveLayers=3,
                              contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    ka, da = detector.detectAndCompute(cv.cvtColor(a, cv.COLOR_BGR2GRAY), masks[0])
    kb, db = detector.detectAndCompute(cv.cvtColor(b, cv.COLOR_BGR2GRAY), masks[1])
    matcher = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    pairs = matcher.knnMatch(da, db, k=2)
    good = [pair[0] for pair in pairs
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    u = np.float32([ka[m.queryIdx].pt for m in good])
    v = np.float32([kb[m.trainIdx].pt for m in good])
    stats = {"keypoints": [len(ka), len(kb)], "knn_pairs": len(pairs),
             "ratio": ratio, "ratio_matches": len(good)}
    return ka, kb, good, u, v, stats


def transform(H, points):
    points = np.asarray(points, dtype=np.float64)
    return cv.perspectiveTransform(points.reshape(1, -1, 2), H)[0]


def outline(image, points, color=(0, 220, 255), width=3):
    cv.polylines(image, [np.rint(points).astype(np.int32)], True, color,
                 width, cv.LINE_AA)


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b)**2, axis=1))))


def point(p):
    return tuple(np.rint(p).astype(int))


def landmark_view(a, b, source, predicted, observed, name):
    canvas = np.concatenate([a, b], axis=1)
    offset = np.array([a.shape[1], 0])
    colors = [(0, 255, 255), (245, 150, 40), (210, 60, 255), (70, 255, 90)]
    for i, (p, q, target) in enumerate(zip(source, predicted, observed)):
        color = colors[i]
        q2, t2 = q + offset, target + offset
        cv.line(canvas, point(p), point(q2), color, 2, cv.LINE_AA)
        cv.circle(canvas, point(p), 5, color, -1)
        cv.circle(canvas, point(q2), 5, color, -1)
        cv.drawMarker(canvas, point(t2), (255, 255, 255), cv.MARKER_CROSS, 14, 2)
    cv.imwrite(str(OUT / name), canvas)


def problem2():
    a, b = (read(f"zen_book_{view}.png") for view in ("frontal", "perspective"))
    corners = np.array([[242, 77], [513, 79], [555, 421], [218, 419]])
    mask_a = polygon_mask(a, corners)
    ka, kb, good, u, v, stats = matches(a, b)
    # Keep the homography focused on the book; feature counts/matches are global.
    book = mask_a[np.rint(u[:, 1]).astype(int), np.rint(u[:, 0]).astype(int)] > 0
    H, status = cv.findHomography(u[book], v[book], cv.RANSAC, 3.0,
                                  maxIters=5000, confidence=0.999)
    H /= H[2, 2]
    inverse = np.linalg.inv(H)
    inliers = np.flatnonzero(book)[status.ravel().astype(bool)]
    warped = b.copy()
    outline(warped, transform(H, corners))
    match_image = cv.drawMatches(a, ka, warped, kb, good, None,
                                 matchColor=(0, 240, 0),
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(str(OUT / "p2_flann.png"), match_image)
    match_image = cv.drawMatches(a, ka, warped, kb, [good[i] for i in inliers], None,
                                 matchColor=(0, 240, 0),
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(str(OUT / "p2_matches.png"), match_image)
    # Independently annotated landmarks: animal eye, hand fingertip, title letters.
    source = np.array([[459, 208], [380, 84], [313, 248]], float)
    observed = np.array([[391, 314], [623, 193], [309, 106]], float)
    reverse_source = np.array([[394, 338], [556, 130], [177, 127]], float)
    reverse_observed = np.array([[477, 208], [329, 100], [332, 329]], float)
    predicted = transform(H, source)
    reverse = transform(inverse, reverse_source)
    landmark_view(a, warped, source, predicted, observed, "p2_forward.png")
    landmark_view(warped, a, reverse_source, reverse, reverse_observed, "p2_inverse.png")
    stats.update({"book_candidates": int(book.sum()), "inliers": len(inliers),
                  "H": H.tolist(), "inverse": inverse.tolist(),
                  "reprojection_rmse": rmse(transform(H, u[inliers]), v[inliers]),
                  "boundary": transform(H, corners).tolist(),
                  "forward": np.c_[source, predicted, observed].tolist(),
                  "reverse": np.c_[reverse_source, reverse, reverse_observed].tolist()})
    return stats


def epipolar_plot(a, b, x1, x2, lines1, lines2, names):
    a, b = a.copy(), b.copy()
    rng = np.random.default_rng(38)
    for p, q, l1, l2 in zip(x1, x2, lines1, lines2):
        color = tuple(map(int, rng.integers(45, 255, 3)))
        for im, position, line in [(a, p, l1), (b, q, l2)]:
            u, v, c = line
            h, w = im.shape[:2]
            if abs(v) >= abs(u):
                start, end = (0, round(-c / v)), (w - 1, round((-c - u * (w - 1)) / v))
            else:
                start, end = (round(-c / u), 0), (round((-c - v * (h - 1)) / u), h - 1)
            ok, start, end = cv.clipLine((0, 0, w, h), start, end)
            if ok:
                cv.line(im, start, end, color, 4, cv.LINE_AA)
            cv.circle(im, point(position), 10, (255, 255, 255), 3, cv.LINE_AA)
            cv.circle(im, point(position), 7, color, -1, cv.LINE_AA)
    panels("p3_epipolar.png", list(zip(names, [a, b])))


def problem3():
    names = ["IMG_4394.jpeg", "IMG_4393.jpeg"]
    a, b = map(read, names)
    # Exclude screen content to avoid a dominant, richly textured plane.
    screens = [([[354, 573], [876, 580], [874, 884], [353, 875]],
                [[902, 604], [1151, 601], [1151, 855], [891, 821]]),
               ([[174, 467], [1030, 485], [1009, 981], [174, 962]],
                [[1070, 533], [1151, 531], [1151, 881], [1042, 852]])]
    masks = []
    views = []
    for im, regions in zip([a, b], screens):
        mask = np.full(im.shape[:2], 255, np.uint8)
        for region in regions:
            cv.fillPoly(mask, [np.int32(region)], 0)
        masks.append(mask)
        visible = im.copy()
        visible[mask == 0] = (58, 58, 58)
        views.append(visible)
    _, _, _, u, v, stats = matches(a, b, masks, ratio=0.75)
    F, status = cv.findFundamentalMat(u, v, cv.USAC_MAGSAC, 1.5, 0.999, 10000)
    inliers = status.ravel().astype(bool)
    u, v = u[inliers], v[inliers]
    F /= np.linalg.norm(F)
    x1, x2 = np.c_[u, np.ones(len(u))], np.c_[v, np.ones(len(v))]
    l1, l2 = x2 @ F, x1 @ F.T
    residual = np.sum(x2 * l2, axis=1)
    sampson = np.abs(residual) / np.sqrt(np.sum(l1[:, :2]**2 + l2[:, :2]**2, axis=1))
    # One point per occupied tile, so the display samples multiple scene areas.
    selected = []
    occupied = set()
    for i in np.argsort(sampson):
        cell = tuple((u[i] // 150).astype(int))
        if cell not in occupied:
            selected.append(i)
            occupied.add(cell)
        if len(selected) == 10:
            break
    epipolar_plot(*views, u[selected], v[selected],
                  cv.computeCorrespondEpilines(v[selected], 2, F).reshape(-1, 3),
                  cv.computeCorrespondEpilines(u[selected], 1, F).reshape(-1, 3), names)
    i = sorted(selected, key=lambda j: sampson[j])[len(selected) // 2]
    stats.update({"inliers": len(u), "F": F.tolist(),
                  "singular_values": np.linalg.svd(F, compute_uv=False).tolist(),
                  "rank": int(np.linalg.matrix_rank(F)),
                  "sample": {"x1": u[i].tolist(), "x2": v[i].tolist(),
                             "residual": float(residual[i]), "sampson_px": float(sampson[i])},
                  "median_sampson_px": float(np.median(sampson)),
                  "line_count": len(selected)})
    return stats


def project(P, points):
    p = np.c_[points, np.ones(len(points))] @ P.T
    return p[:, :2] / p[:, 2, None]


def problem4():
    a, b = [read(f"IMG_{n}.jpeg") for n in (4396, 4398)]
    h, w = a.shape[:2]
    # Approximate intersections of the straight cover edges, excluding the strap.
    cover_a = np.float64([[188, 371], [887, 348], [937, 1278], [182, 1280]])
    cover_b = np.float64([[319, 422], [941, 443], [1053, 1364], [80, 1289]])
    width, height, distance = 4.0, 5.5, 7.0
    dx = np.linalg.norm(cover_a[[1, 2]] - cover_a[[0, 3]], axis=1).mean()
    dy = np.linalg.norm(cover_a[[3, 2]] - cover_a[[0, 1]], axis=1).mean()
    K = np.array([[dx * distance / width, 0, w / 2],
                  [0, dy * distance / height, h / 2], [0, 0, 1]])
    masks = [polygon_mask(a, cover_a), polygon_mask(b, cover_b)]
    for mask, strap in zip(masks, [[[600, 672], [986, 670], [992, 920], [602, 918]],
                                  [[660, 664], [1075, 685], [1080, 956], [651, 909]]]):
        cv.fillPoly(mask, [np.int32(strap)], 0)
    ka, kb, good, u, v, stats = matches(a, b, masks, ratio=0.8)
    H, mask = cv.findHomography(u, v, cv.USAC_MAGSAC, 5.0, maxIters=10000, confidence=0.999)
    H /= H[2, 2]
    ids = np.flatnonzero(mask.ravel())
    # World X right, Y down, Z into the notebook. Origin at the cover center.
    # Use the first view's measured in-plane roll and center offset.
    origin_a = cover_a.mean(axis=0)
    axis_a = np.linalg.solve(K[:2, :2], cover_a[1] - cover_a[0]
                             + cover_a[2] - cover_a[3])
    phi = np.arctan2(axis_a[1], axis_a[0])
    R1 = np.array([[np.cos(phi), -np.sin(phi), 0],
                   [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    t1 = distance * np.linalg.solve(K, np.r_[origin_a, 1.0])
    P1 = K @ np.column_stack((R1, t1))
    plane = H @ P1[:, [0, 1, 3]]
    Q = np.linalg.solve(K, plane)
    scale = 2 / (np.linalg.norm(Q[:, 0]) + np.linalg.norm(Q[:, 1]))
    scale = np.copysign(scale, Q[2, 2])
    r1, r2, t = (Q * scale).T
    U, _, Vt = np.linalg.svd(np.column_stack((r1, r2, np.cross(r1, r2))))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    P2 = K @ np.column_stack((R, t))
    # Factor the recovered projection matrix, with positive diagonal for K.
    Kr, Rr = rq(P2[:, :3])
    signs = np.diag(np.sign(np.diag(Kr)))
    Kr, Rr = Kr @ signs, signs @ Rr
    tr = np.linalg.solve(Kr, P2[:, 3])
    assert np.allclose(Kr @ np.column_stack((Rr, tr)), P2)
    center = -Rr.T @ tr
    assert np.allclose(P2 @ np.r_[center, 1], 0)
    square = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    cube = np.vstack((square, square - [0, 0, 2]))
    a_draw, b_draw = a.copy(), b.copy()
    outline(a_draw, project(P1, square), width=6)
    outline(b_draw, project(P2, square), width=6)
    panels("p4_bases.png", [("Photo A / base", a_draw), ("Photo B / base", b_draw)])
    draw = b.copy()
    uv = project(P2, cube)
    outline(draw, uv[:4], width=6)
    outline(draw, uv[4:], color=(210, 90, 255), width=6)
    for p, q in zip(uv[:4], uv[4:]):
        cv.line(draw, point(p), point(q), (255, 240, 60), 6, cv.LINE_AA)
    cv.imwrite(str(OUT / "p4_cube.png"), draw)
    match_image = cv.drawMatches(a, ka, b, kb, [good[i] for i in ids], None,
                                 matchColor=(0, 240, 0),
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(str(OUT / "p4_matches.png"), match_image)
    planar_a = transform(np.linalg.inv(P1[:, [0, 1, 3]]), u[ids])
    pose_error = rmse(project(P2, np.c_[planar_a, np.zeros(len(ids))]), v[ids])
    stats.update({"image_wh": [w, h], "cover_px": [dx, dy],
                  "width_height_distance_in": [width, height, distance],
                  "K": K.tolist(), "H": H.tolist(), "P1": P1.tolist(), "P2": P2.tolist(),
                  "R": Rr.tolist(), "t": tr.tolist(), "center": center.tolist(),
                  "rq_max_K_diff": float(np.max(np.abs(Kr - K))),
                  "inliers": len(ids), "reprojection_rmse": rmse(transform(H, u[ids]), v[ids]),
                  "pose_rmse": pose_error,
                  "boundary_rmse": rmse(transform(H, cover_a), cover_b)})
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", nargs="?", type=int, choices=range(1, 5))
    args = parser.parse_args()
    functions = [problem1, problem2, problem3, problem4]
    selected = [args.problem] if args.problem else range(1, 5)
    for n in selected:
        cv.setRNGSeed(0)
        result = functions[n - 1]()
        data = json.dumps(result, indent=2)
        (OUT / f"p{n}.json").write_text(data + "\n")
        print(f"Problem {n}\n{data}")
