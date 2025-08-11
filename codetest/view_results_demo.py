"""
Results Analysis Demo - How to View and Analyze Saved Student Data
Demonstrates various ways to analyze the stored learning data
"""

from student_results_manager import StudentResultsManager
import json

def main():
    print("📊 STUDENT RESULTS ANALYSIS DEMO")
    print("=" * 50)
    
    # Initialize results manager
    results_manager = StudentResultsManager()
    
    # Check if any data exists
    try:
        interactions = results_manager.load_json_data(results_manager.interactions_file)
        sessions = results_manager.load_json_data(results_manager.sessions_file)
        evaluations = results_manager.load_json_data(results_manager.evaluations_file)
        
        print(f"📈 Found {len(interactions)} interactions")
        print(f"📈 Found {len(sessions)} sessions")
        print(f"📈 Found {len(evaluations)} evaluations")
        
        if not interactions:
            print("\n⚠️ No data found. Run complete_assignment_demo.py first to generate data.")
            return
        
        print("\n" + "=" * 50)
        print("📊 OVERALL ANALYTICS")
        print("=" * 50)
        
        # Get overall analytics
        analytics = results_manager.get_analytics_summary()
        print(json.dumps(analytics, indent=2))
        
        # Get list of unique students
        unique_students = set()
        for session in sessions:
            if session.get('student_id'):
                unique_students.add(session['student_id'])
        
        print(f"\n📚 Found {len(unique_students)} unique students:")
        for student_id in unique_students:
            print(f"   • {student_id}")
        
        # Generate detailed report for each student
        for student_id in unique_students:
            print(f"\n" + "=" * 60)
            print(f"📋 DETAILED REPORT FOR STUDENT: {student_id}")
            print("=" * 60)
            
            report = results_manager.generate_student_report(student_id)
            print(json.dumps(report, indent=2))
            
            # Export individual student data to CSV
            results_manager.export_to_csv(student_id)
            print(f"✅ Exported {student_id} data to CSV files")
        
        # Export all data to CSV
        print(f"\n📊 Exporting all data to CSV...")
        results_manager.export_to_csv()
        print("✅ All data exported to CSV files")
        
        print(f"\n🎯 ANALYSIS COMPLETE!")
        print(f"📁 All results saved in: {results_manager.storage_dir}")
        print(f"📊 CSV files ready for Excel/data analysis")
        print(f"📈 JSON files ready for programmatic analysis")
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        print("Make sure to run complete_assignment_demo.py first to generate data.")

if __name__ == "__main__":
    main()
