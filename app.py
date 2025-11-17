"""
Polymer Production Scheduler V2 - Modular Version
Main Streamlit application - orchestrates UI and solver.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import streamlit as st
from ortools.sat.python import cp_model

# Import our modular components
from ui.styles import apply_custom_styles, create_header, create_info_box
from ui.components import (
    render_file_uploader,
    render_template_download,
    render_optimization_params,
    render_constraint_info,
    render_data_preview,
    render_results_metrics,
    render_inventory_stats,
    render_welcome_screen
)
from ui.visualizations import (
    get_grade_colors,
    create_gantt_chart,
    create_inventory_chart,
    create_schedule_table,
    create_production_summary
)
from data.loaders import ExcelDataLoader, DataValidator
from core.solver import ProductionScheduler

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Polymer Production Scheduler V2",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
apply_custom_styles()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Header
    create_header(
        "🏭 Polymer Production Scheduler V2",
        "Production scheduling with <strong>strict constraint enforcement</strong>"
    )
    
    # ========================================================================
    # SIDEBAR: File Upload & Parameters
    # ========================================================================
    
    with st.sidebar:
        st.markdown("### 📋 Data Input")
        
        # File upload
        file_bytes = render_file_uploader()
        
        # Template download
        st.markdown("---")
        st.markdown("### 📥 Sample Template")
        render_template_download()
        
        # Parameters (only if file uploaded)
        if file_bytes:
            st.markdown("---")
            st.markdown("### ⚙️ Optimization Parameters")
            params = render_optimization_params()
            
            st.markdown("---")
            render_constraint_info()
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    if file_bytes is None:
        render_welcome_screen()
        return
    
    # ========================================================================
    # LOAD & VALIDATE DATA
    # ========================================================================
    
    try:
        st.markdown("---")
        st.markdown("### 📊 Data Validation")
        
        with st.spinner("Loading data..."):
            loader = ExcelDataLoader(file_bytes)
            plant_df, inventory_df, demand_df, transition_dfs = loader.load_all()
        
        # Validate
        errors, warnings = DataValidator.validate_all(plant_df, inventory_df, demand_df)
        
        if errors:
            for error in errors:
                st.error(error)
            st.stop()
        
        if warnings:
            for warning in warnings:
                st.warning(warning)
        
        st.success("✅ Data validation passed!")
        
        # Display data preview
        st.markdown("---")
        st.markdown("### 📋 Data Preview")
        render_data_preview(plant_df, inventory_df, demand_df)
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        import traceback
        with st.expander("🐛 Debug Info"):
            st.code(traceback.format_exc())
        st.stop()
    
    # ========================================================================
    # OPTIMIZATION
    # ========================================================================
    
    st.markdown("---")
    
    if st.button("🎯 Run Optimization", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize scheduler
            create_info_box("📊 Initializing scheduler...")
            progress_bar.progress(10)
            
            scheduler = ProductionScheduler(
                plant_df, inventory_df, demand_df, transition_dfs,
                params['buffer_days'],
                params['time_limit'],
                params['stockout_penalty'],
                params['transition_penalty']
            )
            
            # Prepare data
            create_info_box("🔧 Preparing data and constraints...")
            progress_bar.progress(20)
            
            constraint_messages = scheduler.prepare_data()
            
            # Display constraint info
            for msg in constraint_messages:
                st.info(msg)
            
            # Build model
            create_info_box("🏗️ Building optimization model...")
            progress_bar.progress(40)
            
            scheduler.build_model()
            
            # Solve
            create_info_box("⚡ Running solver (this may take a few minutes)...")
            progress_bar.progress(60)
            
            status, solve_time = scheduler.solve()
            
            progress_bar.progress(100)
            
            # ================================================================
            # DISPLAY RESULTS
            # ================================================================
            
            st.markdown("---")
            
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                status_msg = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
                create_info_box(f"✅ Optimization completed! Status: {status_msg}", "success")
                
                if scheduler.callback.solutions:
                    solution = scheduler.callback.solutions[-1]
                    
                    st.markdown("### 📊 Optimization Results")
                    
                    # Metrics
                    render_results_metrics(
                        solution['objective'],
                        solution['transitions'],
                        solution['total_stockout'],
                        solve_time
                    )
                    
                    st.markdown("---")
                    
                    # Results tabs
                    tab1, tab2, tab3 = st.tabs([
                        "📅 Production Schedule",
                        "📊 Summary Tables",
                        "📦 Inventory Tracking"
                    ])
                    
                    # ========================================================
                    # TAB 1: PRODUCTION SCHEDULES
                    # ========================================================
                    
                    with tab1:
                        st.markdown("### 📅 Production Schedules by Line")
                        
                        # Generate colors
                        grade_colors = get_grade_colors(scheduler.grades)
                        
                        for line in scheduler.lines:
                            st.markdown(f"#### {line}")
                            
                            # Gantt chart
                            fig = create_gantt_chart(
                                line,
                                solution['schedule'][line],
                                scheduler.dates,
                                scheduler.shutdown_periods.get(line, []),
                                grade_colors
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Schedule table
                            schedule_df = create_schedule_table(
                                line,
                                solution['schedule'][line],
                                scheduler.dates
                            )
                            
                            if not schedule_df.empty:
                                st.dataframe(schedule_df, use_container_width=True)
                            else:
                                st.info(f"No production scheduled for {line}")
                    
                    # ========================================================
                    # TAB 2: SUMMARY TABLES
                    # ========================================================
                    
                    with tab2:
                        st.markdown("### 📊 Production Summary")
                        
                        # Calculate production by (grade, line)
                        prod_by_grade_line = {}
                        for grade in scheduler.grades:
                            for line in scheduler.lines:
                                total = sum(
                                    scheduler.solver.Value(scheduler.production.get((grade, line, d), 0))
                                    for d in range(scheduler.num_days)
                                )
                                prod_by_grade_line[(grade, line)] = total
                        
                        # Calculate total demand by grade
                        demand_by_grade = {}
                        for grade in scheduler.grades:
                            total_demand = sum(
                                scheduler.demand_dict[grade].get(scheduler.dates[d], 0)
                                for d in range(scheduler.num_days - scheduler.buffer_days)
                            )
                            demand_by_grade[grade] = total_demand
                        
                        # Create summary table
                        summary_df = create_production_summary(
                            scheduler.grades,
                            scheduler.lines,
                            prod_by_grade_line,
                            demand_by_grade
                        )
                        
                        st.dataframe(summary_df, use_container_width=True)
                    
                    # ========================================================
                    # TAB 3: INVENTORY TRACKING
                    # ========================================================
                    
                    with tab3:
                        st.markdown("### 📦 Inventory Levels by Grade")
                        
                        for grade in sorted(scheduler.grades):
                            inv_data = [
                                scheduler.solver.Value(scheduler.inventory_vars[(grade, d)])
                                for d in range(scheduler.num_days + 1)
                            ]
                            
                            # Create chart
                            fig = create_inventory_chart(
                                grade,
                                inv_data[:-1],  # Exclude final day
                                scheduler.dates,
                                scheduler.min_inventory[grade],
                                scheduler.max_inventory[grade],
                                scheduler.demand_dict[grade],
                                grade_colors[grade]
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Inventory stats
                            render_inventory_stats(
                                inv_data[0],
                                inv_data[scheduler.num_days - scheduler.buffer_days],
                                max(inv_data),
                                min(inv_data)
                            )
            
            else:
                create_info_box(
                    "⚠️ No feasible solution found. The constraints may be too restrictive. Try:<br>"
                    "• Increasing time limit<br>"
                    "• Relaxing min/max run days<br>"
                    "• Adjusting shutdown periods<br>"
                    "• Increasing opening inventory",
                    "warning"
                )
        
        except Exception as e:
            st.error(f"❌ Optimization error: {str(e)}")
            import traceback
            with st.expander("🐛 Debug Information"):
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()


