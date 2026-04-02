import os
import re
from collections import defaultdict

TARGET_FILE = "FedpuLetterWarm50Bs96Online02.log"
LOG_DIR = "."
OUTPUT_DIR = "reports_letter_proxy_quality"

# 匹配每个 client 行
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
    解析单个日志文件，返回:
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
    """
    读取目录下所有 .log 文件
    返回:
        all_data = {
            filename: {client_id: metric_dict}
        }
    """
    all_data = {}
    for fn in sorted(os.listdir(log_dir)):
        if fn.endswith(".log"):
            path = os.path.join(log_dir, fn)
            parsed = parse_log_file(path)
            if parsed:
                all_data[fn] = parsed
    return all_data


def compare_metric(all_data, target_file, metric):
    """
    对某个指标（ACC/F1/AUC）进行 client-by-client 比较
    返回:
        summary: dict
        winners: list[dict]
        missing_clients: list[int]
    """
    if target_file not in all_data:
        raise FileNotFoundError(f"Target file not found in parsed logs: {target_file}")

    target_clients = set(all_data[target_file].keys())

    # 只比较那些在所有日志里至少出现过、且目标文件里存在的 client
    all_clients_union = set()
    for fn, client_dict in all_data.items():
        all_clients_union.update(client_dict.keys())

    comparable_clients = sorted(target_clients)

    winners = []
    missing_clients = []

    strict_win_count = 0
    tie_win_count = 0
    lose_count = 0

    for client_id in comparable_clients:
        rows = []
        for fn, client_dict in all_data.items():
            if client_id in client_dict:
                rows.append((fn, client_dict[client_id][metric]))

        # 至少得有目标文件和别的文件可比
        if len(rows) <= 1:
            missing_clients.append(client_id)
            continue

        rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
        target_value = all_data[target_file][client_id][metric]
        best_value = rows_sorted[0][1]

        # 找到所有并列第一
        top_files = [fn for fn, val in rows_sorted if abs(val - best_value) < EPS]

        if abs(target_value - best_value) < EPS:
            # 目标文件是第一（可能独占，也可能并列）
            if len(top_files) == 1 and top_files[0] == target_file:
                strict_win = True
                strict_win_count += 1
                tie_status = "Strictly best"
            else:
                strict_win = False
                tie_win_count += 1
                tie_status = "Tied best"

            # 第二名（若目标独占第一，则 second_best 为真正第二；若并列第一，则 second_best 也是第一值）
            if len(rows_sorted) >= 2:
                second_value = rows_sorted[1][1]
            else:
                second_value = rows_sorted[0][1]

            # 找一个非 target 的最强对手
            best_competitor = None
            for fn, val in rows_sorted:
                if fn != target_file:
                    best_competitor = (fn, val)
                    break

            margin = target_value - best_competitor[1] if best_competitor else 0.0

            winners.append({
                "client_id": client_id,
                "target_value": target_value,
                "best_value": best_value,
                "top_files": top_files,
                "tie_status": tie_status,
                "strict_win": strict_win,
                "best_competitor_file": best_competitor[0] if best_competitor else "N/A",
                "best_competitor_value": best_competitor[1] if best_competitor else None,
                "margin_vs_best_competitor": margin,
                "ranking": rows_sorted
            })
        else:
            lose_count += 1

    # 按领先幅度优先，再按目标值降序
    winners.sort(key=lambda x: (x["strict_win"], x["margin_vs_best_competitor"], x["target_value"]), reverse=True)

    summary = {
        "metric": metric,
        "target_file": target_file,
        "num_logs": len(all_data),
        "num_target_clients": len(comparable_clients),
        "strict_win_count": strict_win_count,
        "tie_win_count": tie_win_count,
        "total_best_or_tied_best": strict_win_count + tie_win_count,
        "lose_count": lose_count,
        "missing_or_not_comparable_count": len(missing_clients),
        "all_logs": sorted(all_data.keys())
    }

    return summary, winners, missing_clients


def write_report(output_path, summary, winners, missing_clients):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"Report for metric: {summary['metric']}\n")
        f.write(f"Target file: {summary['target_file']}\n")
        f.write("=" * 100 + "\n\n")

        f.write("[Summary]\n")
        f.write(f"- Number of parsed log files: {summary['num_logs']}\n")
        f.write(f"- Clients found in target file: {summary['num_target_clients']}\n")
        f.write(f"- Strictly best clients: {summary['strict_win_count']}\n")
        f.write(f"- Tied-best clients: {summary['tie_win_count']}\n")
        f.write(f"- Total best-or-tied-best clients: {summary['total_best_or_tied_best']}\n")
        f.write(f"- Lost clients: {summary['lose_count']}\n")
        f.write(f"- Missing / not comparable clients: {summary['missing_or_not_comparable_count']}\n\n")

        f.write("[Parsed log files]\n")
        for name in summary["all_logs"]:
            f.write(f"  - {name}\n")
        f.write("\n")

        f.write("=" * 100 + "\n")
        f.write(f"[Winning clients for {summary['metric']}]\n")
        f.write("=" * 100 + "\n\n")

        if not winners:
            f.write("No winning clients found.\n")
        else:
            for idx, item in enumerate(winners, 1):
                f.write(f"#{idx}\n")
                f.write(f"Client: {item['client_id']:02d}\n")
                f.write(f"Status: {item['tie_status']}\n")
                f.write(f"Target {summary['metric']}: {item['target_value']:.2f}\n")
                if item["best_competitor_value"] is not None:
                    f.write(
                        f"Best competitor: {item['best_competitor_file']} "
                        f"({item['best_competitor_value']:.2f})\n"
                    )
                    f.write(f"Margin vs best competitor: {item['margin_vs_best_competitor']:.2f}\n")
                f.write(f"Top file(s): {', '.join(item['top_files'])}\n")

                f.write("Top-5 ranking:\n")
                for rank, (fn, val) in enumerate(item["ranking"][:5], 1):
                    marker = " <== TARGET" if fn == summary["target_file"] else ""
                    f.write(f"  {rank}. {fn}: {val:.2f}{marker}\n")
                f.write("\n")

        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("[Missing / not comparable clients]\n")
        f.write("=" * 100 + "\n")
        if missing_clients:
            f.write(", ".join(f"{cid:02d}" for cid in missing_clients) + "\n")
        else:
            f.write("None\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = collect_all_logs(LOG_DIR)

    if TARGET_FILE not in all_data:
        raise FileNotFoundError(
            f"Could not parse target file '{TARGET_FILE}' in directory: {os.path.abspath(LOG_DIR)}"
        )

    if len(all_data) < 2:
        raise RuntimeError("Need at least 2 parsed log files for comparison.")

    for metric in METRICS:
        summary, winners, missing_clients = compare_metric(all_data, TARGET_FILE, metric)

        report_path = os.path.join(
            OUTPUT_DIR,
            f"report_{metric.lower()}_for_{TARGET_FILE.replace('.log', '')}.txt"
        )
        write_report(report_path, summary, winners, missing_clients)

        print(f"[OK] {metric} report written to: {report_path}")
        print(
            f"     Strict wins: {summary['strict_win_count']}, "
            f"Tied wins: {summary['tie_win_count']}, "
            f"Total best-or-tied-best: {summary['total_best_or_tied_best']}"
        )

    print(f"\nAll reports saved in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()