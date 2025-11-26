# app.py
# 퍼셉트론 2D 분류 시각화 데모 (Streamlit 버전)

# app.py

import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# 👇 여기 추가
from matplotlib import font_manager, rc
import os

# 레포 루트에 올려둔 폰트 파일 이름
FONT_PATH = os.path.join(os.path.dirname(__file__), "NanumGothic.ttf")

if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    font_prop = font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams["font.family"] = font_prop.get_name()
    # 마이너스 기호 깨지는 것 방지
    plt.rcParams["axes.unicode_minus"] = False
else:
    # 혹시 폰트 파일을 못 찾았을 때 대비
    st.warning("한글 폰트 파일(NanumGothic.ttf)을 찾지 못했습니다. 한글이 깨져 보일 수 있습니다.")



# -----------------------------
# 데이터 생성 함수
# -----------------------------
def generate_data(dataset_type: str, n_per_class: int = 60):
    """
    dataset_type:
      - "easy"  : 두 집단이 멀리 떨어져 있어서 쉽게 분리됨
      - "noisy" : 두 집단이 조금 섞여 있음
      - "xor"   : XOR 패턴 (퍼셉트론이 원래 잘 못 푸는 데이터)
    """
    np.random.seed(None)  # 실행할 때마다 다른 데이터 생성

    if dataset_type == "easy":
        class1 = np.random.randn(n_per_class, 2) * 0.4 + np.array([0.0, 2.0])
        class2 = np.random.randn(n_per_class, 2) * 0.4 + np.array([3.0, -1.0])
        X = np.vstack([class1, class2])
        y = np.hstack([np.ones(n_per_class), -np.ones(n_per_class)])
        name = "쉬운 분류 데이터 (두 집단이 멀리 떨어져 있음)"

    elif dataset_type == "noisy":
        class1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([0.5, 1.5])
        class2 = np.random.randn(n_per_class, 2) * 0.8 + np.array([2.0, 0.0])
        X = np.vstack([class1, class2])
        y = np.hstack([np.ones(n_per_class), -np.ones(n_per_class)])
        name = "노이즈가 있는 데이터 (두 집단이 조금 섞여 있음)"

    else:  # XOR 패턴
        base = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        labels = np.array([1, -1, -1, 1])  # XOR 라벨
        quarter = max(1, n_per_class // 4)

        X_list = []
        y_list = []
        for (corner, lab) in zip(base, labels):
            pts = corner + 0.2 * np.random.randn(quarter, 2)
            X_list.append(pts)
            y_list.append(np.full(quarter, lab))

        X = np.vstack(X_list)
        y = np.hstack(y_list)
        name = "XOR 패턴 데이터 (직선 하나로는 완벽 분리가 어려움)"

    # 시각화 범위
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    return X, y, name, (x_min, x_max, y_min, y_max)


# -----------------------------
# 퍼셉트론 학습 + 시각화 함수
# -----------------------------
def run_perceptron_demo(dataset_type: str, n_per_class: int, lr: float, epochs: int, delay: float):
    # 데이터 준비
    X, y, dataset_name, (x_min, x_max, y_min, y_max) = generate_data(
        dataset_type, n_per_class
    )

    # 퍼셉트론 초기화
    w = np.zeros(2)
    b = 0.0
    errors_per_epoch = []

    plot_placeholder = st.empty()

    for epoch in range(1, epochs + 1):
        errors = 0

        # 샘플 순서 섞기 (학습이 너무 규칙적이지 않게)
        idx = np.random.permutation(len(X))
        X_shuffled = X[idx]
        y_shuffled = y[idx]

        # 퍼셉트론 규칙으로 가중치 업데이트
        for xi, target in zip(X_shuffled, y_shuffled):
            activation = np.dot(w, xi) + b
            if target * activation <= 0:  # 오분류
                w += lr * target * xi
                b += lr * target
                errors += 1

        errors_per_epoch.append(errors)

        # 현재 예측 & 오분류된 점 찾기
        activations = X @ w + b
        y_pred = np.sign(activations)
        misclassified = y_pred != y

        # ---- 그림 생성 ----
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # (왼쪽) 데이터 + 결정 경계
        ax1 = axes[0]
        ax1.set_title(f"{dataset_name}\nEpoch {epoch}/{epochs}, 오분류 {errors}개")

        ax1.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", label="+1 클래스")
        ax1.scatter(X[y == -1, 0], X[y == -1, 1], color="red", label="-1 클래스")

        # 오분류된 점 표시 (검은 테두리)
        if misclassified.any():
            ax1.scatter(
                X[misclassified, 0],
                X[misclassified, 1],
                facecolors="none",
                edgecolors="k",
                s=70,
                linewidths=1.5,
                label="오분류된 점",
            )

        # 결정 경계: w1*x + w2*y + b = 0 → y = -(w1*x + b)/w2
        if w[1] != 0:
            line_x = np.linspace(x_min, x_max, 200)
            line_y = -(w[0] * line_x + b) / w[1]
            ax1.plot(line_x, line_y, color="green", linewidth=2, label="결정 경계")

        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.legend(loc="best")
        ax1.grid(True)

        # (오른쪽) 에폭별 오분류 개수 그래프
        ax2 = axes[1]
        ax2.set_title("에폭(epoch)별 오분류 개수")
        ax2.plot(range(1, epoch + 1), errors_per_epoch, marker="o")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("오분류 개수")
        ax2.grid(True)

        plt.tight_layout()
        plot_placeholder.pyplot(fig)
        plt.close(fig)

        # 너무 빠르게 지나가지 않도록 약간 쉬어주기
        if delay > 0:
            time.sleep(delay)

    # 최종 요약 출력
    st.subheader("학습 요약")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("최종 오분류 개수", f"{errors_per_epoch[-1]}개")
        st.metric("에폭 수", f"{epochs}회")
    with col2:
        st.write("최종 가중치 w:", np.round(w, 3))
        st.write("최종 바이어스 b:", round(b, 3))


# -----------------------------
# Streamlit 메인 함수
# -----------------------------
def main():
    st.set_page_config(
        page_title="퍼셉트론 2D 분류 데모",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 퍼셉트론 2D 분류 시각화 데모")

    st.markdown(
        """
이 웹앱은 **간단한 인공지능 모델인 퍼셉트론(perceptron)**이  
2차원 평면 위의 두 집단(파란 점 / 빨간 점)을 **직선 하나로 어떻게 나누는지** 보여줍니다.
"""
    )

    with st.expander("이 웹앱은 뭐하는 곳인가요? (처음 보는 사람용 간단 설명)", expanded=True):
        st.markdown(
            """
- 왼쪽 그래프:  
  - **파란 점 / 빨간 점** → 서로 다른 두 클래스(두 종류의 데이터)  
  - **초록색 직선** → 퍼셉트론이 학습한 “결정 경계(분리선)”  
  - **검은 테두리 동그라미** → 아직 잘못 분류된 점들  
- 오른쪽 그래프:  
  - 학습이 진행될수록 **오분류(틀린 개수)**가 어떻게 변하는지 보여줍니다.  
- 데이터 모양을 바꿔보면서  
  - “쉽게 나눌 수 있는 데이터”와  
  - “직선 하나로는 잘 안 나누어지는 데이터(XOR)”의 차이를 비교해볼 수 있습니다.
"""
        )

    # ----- 사이드바: 설정 -----
    st.sidebar.header("⚙️ 학습 설정")

    dataset_label = st.sidebar.radio(
        "데이터 모양 선택",
        (
            "쉬운 분류 (멀리 떨어진 두 집단)",
            "노이즈 있는 분류 (조금 섞여 있음)",
            "XOR 패턴 (어려운 예시)",
        ),
    )

    if "쉬운 분류" in dataset_label:
        dataset_type = "easy"
    elif "노이즈" in dataset_label:
        dataset_type = "noisy"
    else:
        dataset_type = "xor"

    n_per_class = st.sidebar.slider(
        "각 클래스당 점 개수 (대략)",
        min_value=20,
        max_value=120,
        value=60,
        step=10,
        help="값이 커질수록 점이 많아지고, 학습에도 조금 더 시간이 걸립니다.",
    )

    lr = st.sidebar.slider(
        "학습률 (learning rate)",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="한 번에 가중치를 얼마나 크게 움직일지 결정하는 값입니다.",
    )

    epochs = st.sidebar.slider(
        "에폭 수 (epochs)",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
        help="데이터 전체를 몇 번 반복해서 학습할지 정합니다.",
    )

    delay = st.sidebar.slider(
        "에폭 사이 대기 시간 (초)",
        min_value=0.0,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="값이 클수록 애니메이션이 느리게 진행됩니다.",
    )

    st.sidebar.markdown("---")
    start = st.sidebar.button("🚀 학습 시작하기")

    st.markdown(
        """
**사용 방법 요약**  
1. 왼쪽 사이드바에서 **데이터 모양, 점 개수, 학습률, 에폭 수**를 정합니다.  
2. `학습 시작하기` 버튼을 누르면, 퍼셉트론이 직선을 조금씩 움직이며  
   두 집단을 나누려고 하는 과정이 애니메이션으로 표시됩니다.  
3. 오른쪽 그래프의 오분류 개수를 보면서  
   - 잘 풀리는 데이터(쉬운 / 노이즈)  
   - 잘 안 풀리는 데이터(XOR)  
   의 차이를 비교해 보세요.
"""
    )

    # ----- 메인 동작 -----
    if start:
        run_perceptron_demo(
            dataset_type=dataset_type,
            n_per_class=n_per_class,
            lr=lr,
            epochs=epochs,
            delay=delay,
        )
    else:
        st.info("왼쪽 사이드바에서 설정을 선택한 뒤 **'학습 시작하기'** 버튼을 눌러보세요.")


if __name__ == "__main__":
    main()
