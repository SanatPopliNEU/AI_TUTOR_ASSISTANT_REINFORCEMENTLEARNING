"""
Quick System Test - Simple verification script
==============================================

This script provides basic checks to see if your system is working.
Run this first before the comprehensive verification.
"""

import os
import sys
from pathlib import Path

def check_basic_setup():
    """Check if the basic project structure exists."""
    print("🔍 CHECKING PROJECT STRUCTURE")
    print("=" * 35)
    
    project_root = Path(__file__).parent.parent
    expected_files = [
        "src/main.py",
        "src/environment/tutoring_environment.py",
        "src/rl/dqn_agent.py",
        "src/rl/ppo_agent.py",
        "src/agents/content_agent.py",
        "src/agents/strategy_agent.py",
        "src/orchestration/tutorial_orchestrator.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in expected_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {missing_files}")
        return False
    else:
        print("\n✅ All required files found!")
        return True

def check_python_packages():
    """Check if we can import basic packages."""
    print("\n🔍 CHECKING PYTHON PACKAGES")
    print("=" * 35)
    
    basic_packages = ['numpy', 'torch', 'matplotlib']
    
    for package in basic_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Install with: pip install {package}")
            return False
    
    print("\n✅ Basic packages available!")
    return True

def main():
    print("🚀 QUICK SYSTEM CHECK")
    print("=" * 25)
    
    structure_ok = check_basic_setup()
    packages_ok = check_python_packages()
    
    if structure_ok and packages_ok:
        print("\n🎉 BASIC CHECKS PASSED!")
        print("\nNext steps:")
        print("1. Run full verification: python scripts/verify_system.py")
        print("2. Or start demo: python src/main.py --mode demo")
    else:
        print("\n❌ SOME CHECKS FAILED!")
        print("Please fix the issues above before proceeding.")

if __name__ == "__main__":
    main()
