"""
Visualization components for production scheduling.
All chart creation logic separated from main app.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Optional


def get_grade_colors(grades: List[str]) -> Dict[str, str]:
    """Generate consistent color mapping for grades."""
    colors = px.colors.qualitative.Vivid + px.colors.qualitative.Safe
    return {grade: colors[i % len(colors)] for i, grade in enumerate(sorted(grades))}


def parse_color_to_rgba(color: str, alpha: float = 1.0) -> str:
    """
    Convert any color format to rgba string.
    
    Args:
        color: Color in hex (#RRGGBB) or rgb(r,g,b) format
        alpha: Alpha channel (0-1)
    
    Returns:
        rgba(r,g,b,a) string
    """
    try:
        if color.startswith('#'):
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
        elif color.startswith('rgb'):
            rgb_values = color.replace('rgb(', '').replace(')', '').split(',')
            r, g, b = [int(x.strip()) for x in rgb_values]
        else:
            r, g, b = 46, 134, 222  # Default blue
        
        return f'rgba({r},{g},{b},{alpha})'
    except:
        return f'rgba(46,134,222,{alpha})'


def create_gantt_chart(
    line: str,
    schedule: List[Optional[str]],
    dates: List[date],
    shutdown_days: List[int],
    grade_colors: Dict[str, str]
) -> Optional[go.Figure]:
    """
    Create Gantt chart for a production line.
    
    Args:
        line: Production line name
        schedule: List of grades by day (None if no production)
        dates: List of dates
        shutdown_days: List of day indices that are shutdowns
        grade_colors: Color mapping for grades
    
    Returns:
        Plotly figure or None if no production
    """
    gantt_data = []
    
    for d, (date_val, grade) in enumerate(zip(dates, schedule)):
        if grade:
            gantt_data.append({
                'Grade': grade,
                'Start': date_val,
                'Finish': date_val + timedelta(days=1),
                'Line': line
            })
    
    if not gantt_data:
        return None
    
    gantt_df = pd.DataFrame(gantt_data)
    
    fig = px.timeline(
        gantt_df,
        x_start='Start',
        x_end='Finish',
        y='Grade',
        color='Grade',
        color_discrete_map=grade_colors,
        title=f'Production Schedule - {line}'
    )
    
    # Add shutdown visualization
    if shutdown_days:
        start_shutdown = dates[shutdown_days[0]]
        end_shutdown = dates[shutdown_days[-1]] + timedelta(days=1)
        
        fig.add_vrect(
            x0=start_shutdown,
            x1=end_shutdown,
            fillcolor='red',
            opacity=0.2,
            layer='below',
            line_width=0,
            annotation_text='Shutdown',
            annotation_position='top left',
            annotation_font_color='red',
            annotation_font_size=12
        )
    
    fig.update_yaxes(autorange='reversed', title=None, showgrid=True)
    fig.update_xaxes(title='Date', dtick='D1', tickformat='%d-%b')
    fig.update_layout(
        height=400,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=150, t=60, b=60)
    )
    
    return fig


def create_inventory_chart(
    grade: str,
    inventory_data: List[float],
    dates: List[date],
    min_inv: float,
    max_inv: float,
    demand_dict: Dict[date, float],
    grade_color: str
) -> go.Figure:
    """
    Create inventory level chart with demand overlay.
    
    Args:
        grade: Grade name
        inventory_data: Daily inventory levels
        dates: List of dates
        min_inv: Minimum inventory threshold
        max_inv: Maximum inventory capacity
        demand_dict: Daily demand by date
        grade_color: Color for this grade
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Inventory area chart
    fill_color = parse_color_to_rgba(grade_color, alpha=0.1)
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=inventory_data,
        mode='lines+markers',
        name='Inventory',
        line=dict(color=grade_color, width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor=fill_color,
        hovertemplate='Date: %{x|%d-%b-%y}<br>Inventory: %{y:.0f} MT<extra></extra>'
    ))
    
    # Demand bars on secondary axis
    demand_values = [demand_dict.get(d, 0) for d in dates]
    
    fig.add_trace(go.Bar(
        x=dates,
        y=demand_values,
        name='Demand',
        marker=dict(color='rgba(255, 100, 100, 0.3)'),
        yaxis='y2',
        hovertemplate='Date: %{x|%d-%b-%y}<br>Demand: %{y:.0f} MT<extra></extra>'
    ))
    
    # Min inventory line
    if min_inv > 0:
        fig.add_hline(
            y=min_inv,
            line=dict(color='red', width=2, dash='dash'),
            annotation_text=f'Min: {min_inv:,.0f}',
            annotation_position='top left',
            annotation_font_color='red'
        )
    
    # Max inventory line
    if max_inv < 1000000:
        fig.add_hline(
            y=max_inv,
            line=dict(color='green', width=2, dash='dash'),
            annotation_text=f'Max: {max_inv:,.0f}',
            annotation_position='bottom left',
            annotation_font_color='green'
        )
    
    fig.update_layout(
        title=f'Inventory Level - {grade}',
        xaxis=dict(title='Date', dtick='D1', tickformat='%d-%b'),
        yaxis=dict(title='Inventory (MT)', side='left'),
        yaxis2=dict(title='Demand (MT)', side='right', overlaying='y'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=450,
        showlegend=True,
        legend=dict(x=1.15, y=1)
    )
    
    return fig


def create_schedule_table(
    line: str,
    schedule: List[Optional[str]],
    dates: List[date]
) -> pd.DataFrame:
    """
    Create tabular schedule showing run periods.
    
    Args:
        line: Production line name
        schedule: List of grades by day
        dates: List of dates
    
    Returns:
        DataFrame with columns: Grade, Start Date, End Date, Days
    """
    schedule_data = []
    current_grade = None
    start_day = None
    
    def format_date(d: date) -> str:
        return d.strftime('%d-%b-%y')
    
    for d, (date_val, grade) in enumerate(zip(dates, schedule)):
        if grade != current_grade:
            # Save previous run
            if current_grade is not None:
                end_date = dates[d - 1]
                duration = (end_date - start_day).days + 1
                schedule_data.append({
                    'Grade': current_grade,
                    'Start Date': format_date(start_day),
                    'End Date': format_date(end_date),
                    'Days': duration
                })
            
            # Start new run
            current_grade = grade
            start_day = date_val
    
    # Add final run
    if current_grade is not None:
        end_date = dates[-1]
        duration = (end_date - start_day).days + 1
        schedule_data.append({
            'Grade': current_grade,
            'Start Date': format_date(start_day),
            'End Date': format_date(end_date),
            'Days': duration
        })
    
    return pd.DataFrame(schedule_data)


def create_production_summary(
    grades: List[str],
    lines: List[str],
    production_by_grade_line: Dict,
    demand_by_grade: Dict
) -> pd.DataFrame:
    """
    Create production summary table.
    
    Args:
        grades: List of grade names
        lines: List of line names
        production_by_grade_line: Dict[(grade, line)] = total_production
        demand_by_grade: Dict[grade] = total_demand
    
    Returns:
        DataFrame with production summary
    """
    summary_data = []
    
    for grade in sorted(grades):
        row = {'Grade': grade}
        grade_total = 0
        
        for line in lines:
            line_prod = production_by_grade_line.get((grade, line), 0)
            row[line] = line_prod
            grade_total += line_prod
        
        row['Total Production'] = grade_total
        
        total_demand = demand_by_grade.get(grade, 0)
        row['Total Demand'] = int(total_demand)
        
        # Service level
        if total_demand > 0:
            service_level = min(100, (grade_total / total_demand) * 100)
            row['Service Level %'] = f"{service_level:.1f}%"
        else:
            row['Service Level %'] = "N/A"
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def create_metrics_html(label: str, value: str, unit: str = "") -> str:
    """
    Create HTML for a metric card.
    
    Args:
        label: Metric label
        value: Metric value
        unit: Optional unit suffix
    
    Returns:
        HTML string
    """
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}{unit}</div>
    </div>
    """
