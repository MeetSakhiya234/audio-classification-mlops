import matplotlib.pyplot as plt

# ==========================================
# WORK BREAKDOWN STRUCTURE
# ==========================================

wbs = [
    ("1.0", "Project Planning"),
    ("2.0", "Dataset Preparation"),
    ("3.0", "Audio Preprocessing"),
    ("4.0", "Wav2Vec2 Model Development"),
    ("5.0", "MLflow Experiment Tracking"),
    ("6.0", "ZenML Pipeline"),
    ("7.0", "ONNX + INT8 Optimization"),
    ("8.0", "FastAPI Development"),
    ("9.0", "Docker Containerization"),
    ("10.0", "Render Deployment"),
    ("11.0", "Prometheus Monitoring"),
    ("12.0", "Grafana Dashboard"),
    ("13.0", "GitHub Actions CI/CD"),
    ("14.0", "Testing and Documentation"),
]

print("=" * 60)
print("WORK BREAKDOWN STRUCTURE")
print("=" * 60)

for number, activity in wbs:
    print(f"{number:<6} {activity}")

print()


# ==========================================
# GANTT CHART DATA
# ==========================================

tasks = [
    ("Project Planning", 1, 1),
    ("Dataset Preparation", 1, 2),
    ("Audio Preprocessing", 2, 2),
    ("Wav2Vec2 Model Development", 2, 3),
    ("Model Evaluation", 3, 2),
    ("MLflow Integration", 4, 2),
    ("ZenML Pipeline", 4, 2),
    ("ONNX Conversion", 5, 1),
    ("INT8 Optimization", 5, 2),
    ("FastAPI Development", 5, 2),
    ("Docker Containerization", 6, 1),
    ("Render Deployment", 6, 2),
    ("Prometheus Monitoring", 7, 1),
    ("Grafana Dashboard", 7, 1),
    ("GitHub Actions CI/CD", 7, 1),
    ("Testing and Documentation", 7, 2),
]


# ==========================================
# CREATE GANTT CHART
# ==========================================

fig, ax = plt.subplots(figsize=(15, 9))

for index, (task, start, duration) in enumerate(tasks):
    ax.barh(
        index,
        duration,
        left=start - 1,
        height=0.5
    )

ax.set_yticks(range(len(tasks)))
ax.set_yticklabels(
    [task[0] for task in tasks],
    fontsize=9
)

ax.set_xticks(range(8))
ax.set_xticklabels([
    "Week 1",
    "Week 2",
    "Week 3",
    "Week 4",
    "Week 5",
    "Week 6",
    "Week 7",
    "Week 8"
])

ax.set_xlim(0, 8)
ax.invert_yaxis()

ax.set_xlabel("Project Timeline")
ax.set_ylabel("Project Activities")

ax.set_title(
    "Gantt Chart - Audio Classification MLOps Project",
    fontsize=16,
    fontweight="bold"
)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    "Gantt_Audio_Classification_MLOps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 60)
print("GANTT CHART CREATED SUCCESSFULLY")
print("Output: Gantt_Audio_Classification_MLOps.png")
print("=" * 60)
