"""
智径 AdaptivePath - 一键启动脚本
====================================
"""
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(cmd, shell=True):
    """运行命令"""
    print(f"\n>>> {' '.join(cmd) if isinstance(cmd, list) else cmd}\n")
    if shell:
        return subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    return subprocess.run(cmd, cwd=PROJECT_ROOT)


def install_deps():
    """安装依赖"""
    print("=" * 70)
    print("  安装依赖")
    print("=" * 70)
    run_command("pip install -r requirements.txt", shell=True)


def run_tests():
    """运行所有测试"""
    print("=" * 70)
    print("  运行测试")
    print("=" * 70)
    for test in ["test_cognitive.py", "test_knowledge.py", "test_agents.py"]:
        print(f"\n--- {test} ---")
        run_command(f"py -3 tests/{test}", shell=True)


def run_cli():
    """启动命令行模式"""
    print("=" * 70)
    print("  启动 CLI 模式")
    print("=" * 70)
    run_command("py -3 src/main.py --mode cli", shell=True)


def run_demo():
    """启动 Web Demo"""
    print("=" * 70)
    print("  启动 Web Demo (Streamlit)")
    print("  访问 http://localhost:8501")
    print("=" * 70)
    run_command("py -3 -m streamlit run demo/app.py", shell=True)


def run_serve():
    """启动 HTTP API"""
    print("=" * 70)
    print("  启动 HTTP API 服务 (FastAPI)")
    print("  访问 http://localhost:8000/docs")
    print("=" * 70)
    run_command("py -3 -m pip install fastapi uvicorn", shell=True)
    run_command("py -3 src/main.py --mode serve --port 8000", shell=True)


def verify():
    """运行系统验证"""
    print("=" * 70)
    print("  系统端到端验证")
    print("=" * 70)
    run_command("py -3 demo/run_demo.py", shell=True)


def main():
    parser = argparse.ArgumentParser(description="智径 AdaptivePath 一键启动")
    parser.add_argument(
        "action",
        choices=["install", "test", "cli", "demo", "serve", "verify", "all"],
        default="demo",
        nargs="?",
    )
    args = parser.parse_args()

    if args.action == "install":
        install_deps()
    elif args.action == "test":
        run_tests()
    elif args.action == "cli":
        run_cli()
    elif args.action == "demo":
        run_demo()
    elif args.action == "serve":
        run_serve()
    elif args.action == "verify":
        verify()
    elif args.action == "all":
        install_deps()
        run_tests()
        verify()
        print("\n✅ All set up! You can now run: py run.py demo")


if __name__ == "__main__":
    main()
