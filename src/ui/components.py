"""
Reusable UI components for Streamlit interface.
Pure UI logic with no business logic.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import io
from typing import Optional, Tuple


def render_file_uploader() -> Optional[bytes]:
    """
    Render file upload widget.
    
    Returns:
        File bytes if uploaded, None otherwise
    """
    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=['xlsx'],
        help="Upload Excel file with Plant, Inventory, and Demand sheets"
    )
    
    if uploaded_file:
        st.success("✅ File uploaded!")
        return uploaded_file.read()
    
    return None


def render_template_download():
    """Render template download button."""
    template = _load_template()
    
    if template:
        st.download_button(
            label="📥 Download Template",
            data=template,
            file_name="polymer_production_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


def _load_template() -> Optional[io.BytesIO]:
    """Load sample template from repository."""
    try:
        for path in [Path("assets/polymer_production_template.xlsx"), 
                     Path("polymer_production_template.xlsx")]:
            if path.exists():
                with open(path, "rb") as f:
                    return io.BytesIO(f.read())
        return None
    except:
        return None


def render_optimization_params() -> dict:
    """
    Render optimization parameter controls.
    
    Returns:
        Dictionary with parameter values
    """
    params = {}
    
    with st.expander("🔧 Solver Settings", expanded=True):
        params['time_limit'] = st.number_input(
            "Time limit (minutes)",
            min_value=1,
            max_value=60,
            value=10,
            help="Maximum time for solver to run"
        )
        
        params['buffer_days'] = st.number_input(
            "Buffer days",
            min_value=0,
            max_value=7,
            value=3,
            help="Additional planning days beyond demand horizon"
        )
    
    with st.expander("🎯 Penalty Weights", expanded=True):
        params['stockout_penalty'] = st.number_input(
            "Stockout penalty",
            min_value=1,
            max_value=1000,
            value=100,
            help="Higher = prioritize meeting demand"
        )
        
        params['transition_penalty'] = st.number_input(
            "Transition penalty",
            min_value=1,
            max_value=100,
            value=10,
            help="Higher = fewer grade changes"
        )
    
    return params


def render_constraint_info():
    """Render information about constraints."""
    st.markdown("""
    <div class="constraint-list">
    <strong>Hard Constraints (Strictly Enforced):</strong><br>
    ✓ Capacity utilization<br>
    ✓ Material running<br>
    ✓ Shutdown periods<br>
    ✓ Min/Max run days<br>
    ✓ Force start dates<br>
    ✓ Allowed lines<br>
    ✓ Rerun allowed<br>
    ✓ Transition rules
    </div>
    """, unsafe_allow_html=True)


def render_data_preview(plant_df: pd.DataFrame, 
                        inventory_df: pd.DataFrame, 
                        demand_df: pd.DataFrame):
    """
    Render data preview tabs.
    
    Args:
        plant_df: Plant data
        inventory_df: Inventory data
        demand_df: Demand data
    """
    tab1, tab2, tab3 = st.tabs(["🏭 Plants", "📦 Inventory", "📈 Demand"])
    
    with tab1:
        st.dataframe(plant_df, use_container_width=True, height=300)
    
    with tab2:
        st.dataframe(inventory_df, use_container_width=True, height=300)
    
    with tab3:
        st.dataframe(demand_df.head(15), use_container_width=True, height=300)


def render_progress_tracker(progress: int, message: str):
    """
    Render progress bar with status message.
    
    Args:
        progress: Progress percentage (0-100)
        message: Status message
    """
    progress_bar = st.progress(progress)
    status_text = st.empty()
    
    from .styles import create_info_box
    status_text.markdown(f'<div class="info-box">{message}</div>', 
                        unsafe_allow_html=True)
    
    return progress_bar, status_text


def render_results_metrics(objective: float, 
                           transitions: int, 
                           stockout: float, 
                           solve_time: float):
    """
    Render result metrics in cards.
    
    Args:
        objective: Objective value
        transitions: Number of transitions
        stockout: Total stockout
        solve_time: Solver time in seconds
    """
    from .styles import create_metric_card
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            create_metric_card("Objective Value", f"{objective:,.0f}"),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            create_metric_card("Transitions", str(transitions)),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            create_metric_card("Total Stockout", f"{stockout:,.0f}", " MT"),
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            create_metric_card("Solve Time", f"{solve_time:.1f}", "s"),
            unsafe_allow_html=True
        )


def render_inventory_stats(opening: float, closing: float, peak: float, minimum: float):
    """
    Render inventory statistics.
    
    Args:
        opening: Opening inventory
        closing: Closing inventory
        peak: Peak inventory
        minimum: Minimum inventory
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Opening", f"{opening:,.0f} MT")
    with col2:
        st.metric("Closing", f"{closing:,.0f} MT")
    with col3:
        st.metric("Peak", f"{peak:,.0f} MT")
    with col4:
        st.metric("Minimum", f"{minimum:,.0f} MT")


def render_welcome_screen():
    """Render welcome screen when no file is uploaded."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        from .styles import create_info_box
        create_info_box(
            "<h3>👈 Get Started</h3>"
            "<p>Upload an Excel file to begin optimization, or download the sample template.</p>",
            "info"
        )
        
        st.markdown("""
        ### ✨ Key Features
        
        - **🎯 Strict Constraint Enforcement**: All hard constraints guaranteed
        - **📊 Interactive Visualizations**: Gantt charts and inventory graphs
        - **🔧 Flexible Configuration**: Adjust penalties and time limits
        - **📈 Real-time Tracking**: Monitor solver progress
        - **🚀 Fast Optimization**: Powered by Google OR-Tools CP-SAT
        
        ### 📋 Hard Constraints (Always Enforced)
        
        1. Full capacity utilization
        2. Material running for specified initial days
        3. Expected run days for current materials
        4. Shutdown periods - zero production during shutdowns
        5. Opening inventory - exact starting levels
        6. Min & Max run days - consecutive production limits
        7. Force start dates - mandatory production starts
        8. Allowed lines - grade-plant restrictions
        9. Rerun allowed - single vs. multiple runs
        10. Transition rules - forbidden grade changes
        
        ### 📊 Soft Constraints (Optimized)
        
        - Minimum inventory levels (penalized if violated)
        - Minimum closing inventory (penalized if not met)
        - Transition minimization (via penalty weights)
        - Stockout prevention (primary objective)
        """)
    
    with col2:
        st.markdown("""
        ### 📖 Quick Guide
        
        **1. Prepare Excel File**
        - Plant sheet: capacities, shutdowns
        - Inventory sheet: constraints per grade
        - Demand sheet: daily requirements
        - Transition sheets: allowed changes
        
        **2. Upload & Configure**
        - Upload your Excel file
        - Set time limit
        - Adjust penalty weights
        
        **3. Optimize**
        - Click "Run Optimization"
        - Monitor progress
        - Review results
        
        **4. Analyze**
        - View Gantt charts
        - Check inventory levels
        - Export schedules
        """)
