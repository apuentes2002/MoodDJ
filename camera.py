import cv2
import math
from collections import deque
import mediapipe as mp


# ------------- Landmark helpers -------------
def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def xy(lmk, wi, hi):
    return int(lmk.x * wi), int(lmk.y * hi)


# Face Mesh landmark indices (MediaPipe)
# Mouth
LM_MOUTH_LEFT = 61
LM_MOUTH_RIGHT = 291
LM_LIP_UP = 13
LM_LIP_DOWN = 14

# Eyes
R_EYE_OUTER = 33
R_EYE_INNER = 133
R_EYE_UP = 159
R_EYE_DOWN = 145

L_EYE_OUTER = 263
L_EYE_INNER = 362
L_EYE_UP = 386
L_EYE_DOWN = 374

# Eyebrows - multiple points for curved lines
# Right eyebrow (from inner to outer)
RBROW_INNER = 70
RBROW_MID1 = 63
RBROW_MID2 = 105
RBROW_MID3 = 66
RBROW_OUTER = 107

# Left eyebrow (from inner to outer)
LBROW_INNER = 300
LBROW_MID1 = 293
LBROW_MID2 = 334
LBROW_MID3 = 296
LBROW_OUTER = 336

# ------------- Smoothing -------------
WIN = 10  # frames to smooth over
hist_smile = deque(maxlen=WIN)
hist_mouth_open = deque(maxlen=WIN)
hist_eye_open = deque(maxlen=WIN)
hist_brow_raise = deque(maxlen=WIN)
hist_brow_distance = deque(maxlen=WIN)

# ------------- Thresholds (tune if needed) -------------
SMILE_WIDTH_MIN = 0.60  # mouth width vs inter-ocular width
MOUTH_OPEN_SURPRISE = 0.20  # mouth openness vs inter-ocular width
EYE_OPEN_SURPRISE = 0.27  # average eyelid gap vs eye width
BROW_RAISE_SURPRISE = 0.20  # eyebrow height vs eye height

# Angry thresholds
BROW_LOWERED_ANGRY = 0.18  # Lowered brows
BROW_CLOSE_ANGRY = 0.36  # Brows close together
EYE_SQUINT_ANGRY = 0.22  # Eyes squinting/narrowed
MOUTH_NEUTRAL_MAX = 0.55  # Mouth not smiling

# ------------- MediaPipe setup -------------
mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
            p_ml = xy(lm[LM_MOUTH_LEFT], w, h)
            p_mr = xy(lm[LM_MOUTH_RIGHT], w, h)
            p_mu = xy(lm[LM_LIP_UP], w, h)
            p_md = xy(lm[LM_LIP_DOWN], w, h)

            p_re_outer = xy(lm[R_EYE_OUTER], w, h)
            p_re_inner = xy(lm[R_EYE_INNER], w, h)
            p_re_up = xy(lm[R_EYE_UP], w, h)
            p_re_down = xy(lm[R_EYE_DOWN], w, h)

            p_le_outer = xy(lm[L_EYE_OUTER], w, h)
            p_le_inner = xy(lm[L_EYE_INNER], w, h)
            p_le_up = xy(lm[L_EYE_UP], w, h)
            p_le_down = xy(lm[L_EYE_DOWN], w, h)

            p_rbrow_inner = xy(lm[RBROW_INNER], w, h)
            p_rbrow_mid1 = xy(lm[RBROW_MID1], w, h)
            p_rbrow_mid2 = xy(lm[RBROW_MID2], w, h)
            p_rbrow_mid3 = xy(lm[RBROW_MID3], w, h)
            p_rbrow_outer = xy(lm[RBROW_OUTER], w, h)

            p_lbrow_inner = xy(lm[LBROW_INNER], w, h)
            p_lbrow_mid1 = xy(lm[LBROW_MID1], w, h)
            p_lbrow_mid2 = xy(lm[LBROW_MID2], w, h)
            p_lbrow_mid3 = xy(lm[LBROW_MID3], w, h)
            p_lbrow_outer = xy(lm[LBROW_OUTER], w, h)

            # --- Base scales ---
            inter_ocular = dist(p_re_outer, p_le_outer) + 1e-6
            re_width = dist(p_re_outer, p_re_inner) + 1e-6
            le_width = dist(p_le_outer, p_le_inner) + 1e-6
            eye_width_avg = (re_width + le_width) / 2.0

            # --- Features ---
            mouth_width = dist(p_ml, p_mr) / inter_ocular
            mouth_open = dist(p_mu, p_md) / inter_ocular

            re_open = dist(p_re_up, p_re_down) / re_width
            le_open = dist(p_le_up, p_le_down) / le_width
            eye_open = (re_open + le_open) / 2.0

            # Eyebrow height (using middle points)
            re_center = ((p_re_outer[0] + p_re_inner[0]) // 2, (p_re_outer[1] + p_re_inner[1]) // 2)
            le_center = ((p_le_outer[0] + p_le_inner[0]) // 2, (p_le_outer[1] + p_le_inner[1]) // 2)
            rbrow_raise = abs(p_rbrow_mid2[1] - re_center[1]) / inter_ocular
            lbrow_raise = abs(p_lbrow_mid2[1] - le_center[1]) / inter_ocular
            brow_raise = (rbrow_raise + lbrow_raise) / 2.0

            # Brow closeness (using inner points)
            brow_distance = dist(p_rbrow_inner, p_lbrow_inner) / inter_ocular

            # --- Smooth features ---
            hist_smile.append(mouth_width)
            hist_mouth_open.append(mouth_open)
            hist_eye_open.append(eye_open)
            hist_brow_raise.append(brow_raise)
            hist_brow_distance.append(brow_distance)

            s_smile = sum(hist_smile) / len(hist_smile)
            s_mopen = sum(hist_mouth_open) / len(hist_mouth_open)
            s_eopen = sum(hist_eye_open) / len(hist_eye_open)
            s_brow = sum(hist_brow_raise) / len(hist_brow_raise)
            s_brow_dist = sum(hist_brow_distance) / len(hist_brow_distance)

            # --- Mood rules ---
            # SURPRISED: either full surprise or eyebrow-only
            if ((s_mopen > MOUTH_OPEN_SURPRISE and
                 s_eopen > EYE_OPEN_SURPRISE and
                 s_brow > BROW_RAISE_SURPRISE) or
                    (s_brow > 0.30)):
                mood = "surprised"

            # HAPPY: wide mouth (smile)
            elif s_smile > SMILE_WIDTH_MIN:
                mood = "happy"

            else:
                # ANGRY: Requires ALL conditions to be met
                # Eye squinting + neutral mouth + lowered eyebrows
                eyes_squinting = s_eopen < 0.20  # Eyes narrowed/squinting (more sensitive)
                mouth_neutral = s_smile < 0.55 and s_mopen < 0.15  # Not smiling, not wide open
                brows_lowered = s_brow < 0.24  # Eyebrows lowered below neutral (more sensitive)

                # Anger if: ALL THREE conditions are met
                if eyes_squinting and mouth_neutral and brows_lowered:
                    mood = "angry"

                # SAD: mouth not wide + lips not open
                elif s_smile < 0.42 and s_mopen < 0.14:
                    mood = "sad"

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

            # Brows - curved lines with multiple points for each eyebrow
            # Right eyebrow curve
            cv2.line(frame, p_rbrow_inner, p_rbrow_mid1, (0, 150, 255), 2)
            cv2.line(frame, p_rbrow_mid1, p_rbrow_mid2, (0, 150, 255), 2)
            cv2.line(frame, p_rbrow_mid2, p_rbrow_mid3, (0, 150, 255), 2)
            cv2.line(frame, p_rbrow_mid3, p_rbrow_outer, (0, 150, 255), 2)

            # Left eyebrow curve
            cv2.line(frame, p_lbrow_inner, p_lbrow_mid1, (0, 150, 255), 2)
            cv2.line(frame, p_lbrow_mid1, p_lbrow_mid2, (0, 150, 255), 2)
            cv2.line(frame, p_lbrow_mid2, p_lbrow_mid3, (0, 150, 255), 2)
            cv2.line(frame, p_lbrow_mid3, p_lbrow_outer, (0, 150, 255), 2)

            # Eyebrow points
            for pt in (p_rbrow_inner, p_rbrow_mid1, p_rbrow_mid2, p_rbrow_mid3, p_rbrow_outer):
                cv2.circle(frame, pt, 3, (0, 200, 255), -1)
            for pt in (p_lbrow_inner, p_lbrow_mid1, p_lbrow_mid2, p_lbrow_mid3, p_lbrow_outer):
                cv2.circle(frame, pt, 3, (0, 200, 255), -1)

            # Angry parameters - ALWAYS visible (top-left)
            cv2.rectangle(frame, (8, 8), (320, 155), (0, 0, 0), -1)
            cv2.putText(frame, "ANGRY DETECTION:", (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, f"Eye squint: {s_eopen:.2f} < 0.20",
                        (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if s_eopen < 0.20 else (0, 255, 255), 1)
            cv2.putText(frame, f"Mouth neutral: {s_smile:.2f} < 0.55",
                        (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if s_smile < 0.55 else (0, 255, 255), 1)
            cv2.putText(frame, f"Mouth closed: {s_mopen:.2f} < 0.15",
                        (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if s_mopen < 0.15 else (0, 255, 255), 1)
            cv2.putText(frame, f"Brows lowered: {s_brow:.2f} < 0.24",
                        (16, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if s_brow < 0.24 else (0, 255, 255), 1)
            cv2.putText(frame, f"Status: {'ANGRY' if mood == 'angry' else 'Not Angry'}",
                        (16, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 0) if mood == "angry" else (100, 100, 100), 1)

            # Happy parameters - ALWAYS visible (left side, below angry)
            cv2.rectangle(frame, (8, 165), (320, 235), (0, 0, 0), -1)
            cv2.putText(frame, "HAPPY DETECTION:", (16, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, f"Smile width: {s_smile:.2f} > 0.60",
                        (16, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if s_smile > 0.60 else (0, 255, 255), 1)
            cv2.putText(frame, f"Status: {'HAPPY' if mood == 'happy' else 'Not Happy'}",
                        (16, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 0) if mood == "happy" else (100, 100, 100), 1)

            # Surprised parameters - ALWAYS visible (left side, below happy)
            cv2.rectangle(frame, (8, 245), (320, 355), (0, 0, 0), -1)
            cv2.putText(frame, "SURPRISED DETECTION:", (16, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, f"Mouth open: {s_mopen:.2f} > 0.20",
                        (16, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if s_mopen > 0.20 else (0, 255, 255), 1)
            cv2.putText(frame, f"Eyes wide: {s_eopen:.2f} > 0.27",
                        (16, 307), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if s_eopen > 0.27 else (0, 255, 255), 1)
            cv2.putText(frame, f"Brows raised: {s_brow:.2f} > 0.20",
                        (16, 329), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if s_brow > 0.20 else (0, 255, 255), 1)
            cv2.putText(frame, f"Status: {'SURPRISED' if mood == 'surprised' else 'Not Surprised'}",
                        (16, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 0) if mood == "surprised" else (100, 100, 100), 1)

            # Sad parameters - ALWAYS visible (left side, below surprised)
            cv2.rectangle(frame, (8, 365), (320, 455), (0, 0, 0), -1)
            cv2.putText(frame, "SAD DETECTION:", (16, 382), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, f"Smile narrow: {s_smile:.2f} < 0.42",
                        (16, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if s_smile < 0.42 else (0, 255, 255), 1)
            cv2.putText(frame, f"Mouth closed: {s_mopen:.2f} < 0.14",
                        (16, 427), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if s_mopen < 0.14 else (0, 255, 255), 1)
            cv2.putText(frame, f"Status: {'SAD' if mood == 'sad' else 'Not Sad'}",
                        (16, 448), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 0) if mood == "sad" else (100, 100, 100), 1)

            # All raw parameters window - ALWAYS visible (top-right)
            cv2.rectangle(frame, (w - 380, 8), (w - 8, 180), (0, 0, 0), -1)
            cv2.putText(frame, "RAW PARAMETERS:", (w - 370, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"smile_w: {s_smile:.3f}", (w - 360, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(frame, f"mouth_open: {s_mopen:.3f}", (w - 360, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)
            cv2.putText(frame, f"eye_open: {s_eopen:.3f}", (w - 360, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
            cv2.putText(frame, f"brow_raise: {s_brow:.3f}", (w - 360, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 200, 255), 1)
            cv2.putText(frame, f"brow_dist: {s_brow_dist:.3f}", (w - 360, 153), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (150, 150, 255), 1)
            cv2.putText(frame, f"MOOD: {mood.upper()}", (w - 360, 173), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Mood banner with color coding
        if mood == "happy":
            color = (0, 200, 0)
        elif mood == "surprised":
            color = (0, 200, 255)
        elif mood == "angry":
            color = (0, 0, 255)  # Red for angry
        elif mood == "sad":
            color = (255, 100, 0)
        else:
            color = (160, 160, 160)

        cv2.rectangle(frame, (8, h - 50), (180, h - 12), (0, 0, 0), -1)
        cv2.putText(frame, f"{mood.upper()}", (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Mood DJ - Squinting Anger Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()