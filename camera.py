import cv2
import math
from collections import deque
import mediapipe as mp

# ------------- Landmark helpers -------------
def dist(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1])

def xy(lmk, wi, hi):
    return int(lmk.x * wi), int(lmk.y * hi)

# Face Mesh landmark indices (MediaPipe)
# Mouth
LM_MOUTH_LEFT  = 61
LM_MOUTH_RIGHT = 291
LM_LIP_UP      = 13
LM_LIP_DOWN    = 14

# Eyes (RIGHT eye uses indices 33, 133 (corners), 159 (upper lid), 145 (lower lid))
#      LEFT  eye uses indices 263, 362 (corners), 386 (upper lid), 374 (lower lid))
R_EYE_OUTER = 33
R_EYE_INNER = 133
R_EYE_UP    = 159
R_EYE_DOWN  = 145

L_EYE_OUTER = 263
L_EYE_INNER = 362
L_EYE_UP    = 386
L_EYE_DOWN  = 374

# Eyebrow "mid" points (approx.)
# RIGHT brow: 105; LEFT brow: 334 (above eye centers)
RBROW = 105
LBROW = 334

# ------------- Smoothing -------------
WIN = 10  # frames to smooth over
hist_smile = deque(maxlen=WIN)
hist_mouth_open = deque(maxlen=WIN)
hist_eye_open = deque(maxlen=WIN)
hist_brow_raise = deque(maxlen=WIN)

# ------------- Thresholds (tune if needed) -------------
# All features are normalized, so these work for most faces/cameras.
SMILE_WIDTH_MIN = 0.38   # mouth width vs inter-ocular width
MOUTH_OPEN_SURPRISE = 0.20  # mouth openness vs inter-ocular width
EYE_OPEN_SURPRISE   = 0.27  # average eyelid gap vs eye width
BROW_RAISE_SURPRISE = 0.20  # eyebrow height vs eye height

# ------------- MediaPipe setup -------------
mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW helps startup on Windows
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try index 1 or close other apps using the camera.")

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror view
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        faces = getattr(results, "multi_face_landmarks", None)

        mood = "neutral"

        if faces:
            lm = faces[0].landmark

            # --- Key points (pixels) ---
            p_ml = xy(lm[LM_MOUTH_LEFT],  w, h)
            p_mr = xy(lm[LM_MOUTH_RIGHT], w, h)
            p_mu = xy(lm[LM_LIP_UP],      w, h)
            p_md = xy(lm[LM_LIP_DOWN],    w, h)

            p_re_outer = xy(lm[R_EYE_OUTER], w, h)
            p_re_inner = xy(lm[R_EYE_INNER], w, h)
            p_re_up    = xy(lm[R_EYE_UP],    w, h)
            p_re_down  = xy(lm[R_EYE_DOWN],  w, h)

            p_le_outer = xy(lm[L_EYE_OUTER], w, h)
            p_le_inner = xy(lm[L_EYE_INNER], w, h)
            p_le_up    = xy(lm[L_EYE_UP],    w, h)
            p_le_down  = xy(lm[L_EYE_DOWN],  w, h)

            p_rbrow = xy(lm[RBROW], w, h)
            p_lbrow = xy(lm[LBROW], w, h)

            # --- Base scales ---
            inter_ocular = dist(p_re_outer, p_le_outer) + 1e-6  # across-face scale
            re_width = dist(p_re_outer, p_re_inner) + 1e-6
            le_width = dist(p_le_outer, p_le_inner) + 1e-6
            eye_width_avg = (re_width + le_width) / 2.0

            # --- Features ---
            mouth_width = dist(p_ml, p_mr) / inter_ocular
            mouth_open  = dist(p_mu, p_md) / inter_ocular

            re_open = dist(p_re_up, p_re_down) / re_width
            le_open = dist(p_le_up, p_le_down) / le_width
            eye_open = (re_open + le_open) / 2.0  # eyelid gap normalized by eye width

            # Eyebrow height vs eye "height" (use eyelid gap as a proxy for eye height)
            # Measure brow distance from the eye's centerline (average of corners)
            re_center = ((p_re_outer[0]+p_re_inner[0])//2, (p_re_outer[1]+p_re_inner[1])//2)
            le_center = ((p_le_outer[0]+p_le_inner[0])//2, (p_le_outer[1]+p_le_inner[1])//2)
            # Vertical (y) distance from eye centers to brows; normalize by inter-ocular
            rbrow_raise = abs(p_rbrow[1] - re_center[1]) / inter_ocular
            lbrow_raise = abs(p_lbrow[1] - le_center[1]) / inter_ocular
            brow_raise = (rbrow_raise + lbrow_raise) / 2.0

            # --- Smooth features ---
            hist_smile.append(mouth_width)
            hist_mouth_open.append(mouth_open)
            hist_eye_open.append(eye_open)
            hist_brow_raise.append(brow_raise)

            s_smile = sum(hist_smile)/len(hist_smile)
            s_mopen = sum(hist_mouth_open)/len(hist_mouth_open)
            s_eopen = sum(hist_eye_open)/len(hist_eye_open)
            s_brow  = sum(hist_brow_raise)/len(hist_brow_raise)

            # --- Mood rules (simple, readable) ---
            # SURPRISED: big mouth open + wide eyes + raised brows
            if (s_mopen > MOUTH_OPEN_SURPRISE and
                s_eopen > EYE_OPEN_SURPRISE and
                s_brow  > BROW_RAISE_SURPRISE):
                mood = "surprised"

            # HAPPY: wide mouth (smile). Allow eyes to be normal or slightly squinty.
            elif s_smile > SMILE_WIDTH_MIN:
                mood = "happy"

            else:
                mood = "neutral"

            # --- Visualization ---
            # Mouth line + lip points
            cv2.line(frame, p_ml, p_mr, (0, 255, 0), 1)
            for pt in (p_ml, p_mr, p_mu, p_md):
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

            # Eye markers
            for pt in (p_re_outer, p_re_inner, p_re_up, p_re_down,
                       p_le_outer, p_le_inner, p_le_up, p_le_down):
                cv2.circle(frame, pt, 2, (255, 255, 255), -1)

            # Brows
            cv2.circle(frame, p_rbrow, 2, (0, 200, 255), -1)
            cv2.circle(frame, p_lbrow, 2, (0, 200, 255), -1)

            # Feature HUD
            #cv2.rectangle(frame, (8, 8), (360, 90), (0, 0, 0), -1)
            #cv2.putText(frame, f"smile_w: {s_smile:.3f}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            #cv2.putText(frame, f"mouth_open: {s_mopen:.3f}", (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
            #cv2.putText(frame, f"eye_open: {s_eopen:.3f}", (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        # Mood banner
        color = (0, 200, 0) if mood == "happy" else (0, 200, 255) if mood == "surprised" else (160, 160, 160)
        cv2.rectangle(frame, (8, h-50), (180, h-12), (0,0,0), -1)
        cv2.putText(frame, f"{mood.upper()}", (16, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Mood DJ - Mouth/Eyebrows/Eyes", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()