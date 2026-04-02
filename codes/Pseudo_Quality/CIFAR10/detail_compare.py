import os
import re

TARGET_FILE = "FedpuCIFARWarm50Bs16Weight05Online02.log"
LOG_DIR = "."
OUTPUT_DIR = "reports_clientwise_win_count"

LINE_PATTERN = re.compile(
    r"\[Client\s+(\d+)\].*?"
    r"ACC:\s*([\d.]+)\s*\|\s*"
    r"Pre:\s*([\d.]+)\s*\|\s*"
    r"Rec:\s*([\d.]+)\s*\|\s*"
    r"F1:\s*([\d.]+)\s*\|\s*"
    r"AUC:\s*([\d.]+)"
)

METRICS = ["ACC", "F1", "AUC"]
EPS = 1e-9


def parse_log_file(filepath):
    """
    返回:
    {
        client_id: {
            "ACC": float,
            "F1": float,
            "AUC": float,
            "raw_line": str
        }
    }
    """
    result = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_PATTERN.search(line)
            if m:
                client_id = int(m.group(1))
                acc = float(m.group(2))
                f1 = float(m.group(5))
                auc = float(m.group(6))
                result[client_id] = {
                    "ACC": acc,
                    "F1": f1,
                    "AUC": auc,
                    "raw_line": line.strip()
                }
    return result


def collect_all_logs(log_dir):
    all_data = {}
    for fn in sorted(os.listdir(log_dir)):
        if fn.endswith(".log"):
            path = os.path.join(log_dir, fn)
            parsed = parse_log_file(path)
            if parsed:
                all_data[fn] = parsed
    return all_data


def analyze_metric(all_data, target_file, metric):
    """
    对一个 metric 做逐 client 分析:
    - 赢了多少个 baseline
    - 输给哪些 baseline
    - 打平哪些 baseline
    """
    if target_file not in all_data:
        raise FileNotFoundError(f"Target file not found: {target_file}")

    baselines = [fn for fn in sorted(all_data.keys()) if fn != target_file]
    target_clients = sorted(all_data[target_file].keys())

    results = []

    for client_id in target_clients:
        if client_id not in all_data[target_file]:
            continue

        target_value = all_data[target_file][client_id][metric]

        win_list = []
        lose_list = []
        tie_list = []
        missing_list = []

        for baseline in baselines:
            if client_id not in all_data[baseline]:
                missing_list.append(baseline)
                continue

            base_value = all_data[baseline][client_id][metric]

            if target_value > base_value + EPS:
                win_list.append((baseline, base_value))
            elif target_value < base_value - EPS:
                lose_list.append((baseline, base_value))
            else:
                tie_list.append((baseline, base_value))

        results.append({
            "client_id": client_id,
            "target_value": target_value,
            "win_count": len(win_list),
            "lose_count": len(lose_list),
            "tie_count": len(tie_list),
            "missing_count": len(missing_list),
            "win_list": sorted(win_list, key=lambda x: x[1]),
            "lose_list": sorted(lose_list, key=lambda x: x[1], reverse=True),
            "tie_list": sorted(tie_list, key=lambda x: x[0]),
            "missing_list": sorted(missing_list),
        })

    # 排序规则：
    # 1. 赢的 baseline 数越多越靠前
    # 2. 输的越少越靠前
    # 3. target 指标值越大越靠前
    # 4. client id 越小越靠前
    results.sort(
        key=lambda x: (-x["win_count"], x["lose_count"], -x["target_value"], x["client_id"])
    )

    return results, baselines


def write_report(output_path, metric, target_file, results, baselines):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 110 + "\n")
        f.write(f"Client-wise win/loss report for metric: {metric}\n")
        f.write(f"Target file: {target_file}\n")
        f.write("=" * 110 + "\n\n")

        f.write("[Compared baselines]\n")
        for b in baselines:
            f.write(f"  - {b}\n")
        f.write(f"\nTotal baselines: {len(baselines)}\n\n")

        f.write("=" * 110 + "\n")
        f.write("[Client ranking]\n")
        f.write("=" * 110 + "\n\n")

        for rank, item in enumerate(results, 1):
            f.write(f"#{rank}  Client {item['client_id']:02d}\n")
            f.write(f"Target {metric}: {item['target_value']:.2f}\n")
            f.write(
                f"Beat {item['win_count']}/{len(baselines)} baselines | "
                f"Lose to {item['lose_count']} | "
                f"Tie with {item['tie_count']} | "
                f"Missing in {item['missing_count']}\n"
            )

            # 赢了哪些 baseline
            if item["win_list"]:
                f.write("Beat baselines:\n")
                for name, value in item["win_list"]:
                    margin = item["target_value"] - value
                    f.write(f"  - {name}: {value:.2f}  (margin +{margin:.2f})\n")
            else:
                f.write("Beat baselines:\n")
                f.write("  - None\n")

            # 输给哪些 baseline
            if item["lose_list"]:
                f.write("Lost to baselines:\n")
                for name, value in item["lose_list"]:
                    margin = item["target_value"] - value
                    f.write(f"  - {name}: {value:.2f}  (margin {margin:.2f})\n")
            else:
                f.write("Lost to baselines:\n")
                f.write("  - None\n")

            # 打平
            if item["tie_list"]:
                f.write("Tied baselines:\n")
                for name, value in item["tie_list"]:
                    f.write(f"  - {name}: {value:.2f}\n")
            else:
                f.write("Tied baselines:\n")
                f.write("  - None\n")

            # 缺失
            if item["missing_list"]:
                f.write("Missing baselines:\n")
                for name in item["missing_list"]:
                    f.write(f"  - {name}\n")
            else:
                f.write("Missing baselines:\n")
                f.write("  - None\n")

            f.write("-" * 110 + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = collect_all_logs(LOG_DIR)
    if TARGET_FILE not in all_data:
        raise FileNotFoundError(f"Could not parse target file: {TARGET_FILE}")

    if len(all_data) < 2:
        raise RuntimeError("Need at least 2 parsed log files.")

    for metric in METRICS:
        results, baselines = analyze_metric(all_data, TARGET_FILE, metric)
        report_path = os.path.join(
            OUTPUT_DIR,
            f"clientwise_winloss_{metric.lower()}_{TARGET_FILE.replace('.log', '')}.txt"
        )
        write_report(report_path, metric, TARGET_FILE, results, baselines)
        print(f"[OK] {metric} report saved to: {report_path}")

    print(f"\nAll reports saved in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()