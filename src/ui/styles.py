"""
Styling and CSS for the application.
All visual styling separated from logic.
"""

# Theme colors
THEME = {
    'primary': '#2E86DE',
    'secondary': '#54A0FF',
    'success': '#26DE81',
    'warning': '#FED330',
    'danger': '#FC5C65',
    'neutral': {
        'bg': '#F8F9FA',
        'border': '#DEE2E6',
        'text': '#495057',
        'text_light': '#6C757D'
    }
}

# Custom CSS for Streamlit
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E86DE;
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border-bottom: 3px solid #2E86DE;
    }
    
    .sub-header {
        text-align: center;
        color: #6C757D;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #DEE2E6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2E86DE;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6C757D;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background: #D1ECF1;
        border: 1px solid #BEE5EB;
        color: #0C5460;
        margin: 1rem 0;
    }
    
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background: #D4EDDA;
        border: 1px solid #C3E6CB;
        color: #155724;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .warning-box {
        padding: 1rem;
        border-radius: 8px;
        background: #FFF3CD;
        border: 1px solid #FFEAA7;
        color: #856404;
        margin: 1rem 0;
    }
    
    .error-box {
        padding: 1rem;
        border-radius: 8px;
        background: #F8D7DA;
        border: 1px solid #F5C6CB;
        color: #721C24;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: #2E86DE;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background: #1E5FBD;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(46, 134, 222, 0.3);
    }
    
    .constraint-list {
        font-size: 0.85rem;
        color: #6C757D;
        line-height: 1.6;
    }
    
    .constraint-list strong {
        color: #2E86DE;
    }
    
    /* Improve dataframe appearance */
    .dataframe {
        border: 1px solid #DEE2E6 !important;
        border-radius: 8px;
    }
    
    /* Better tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F8F9FA;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        background-color: white;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2E86DE;
        color: white;
    }
</style>
"""


def apply_custom_styles():
    """Apply custom CSS to Streamlit app."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def create_header(title: str, subtitle: str = None):
    """Create styled header."""
    import streamlit as st
    
    header_html = f'<div class="main-header">{title}</div>'
    st.markdown(header_html, unsafe_allow_html=True)
    
    if subtitle:
        sub_html = f'<div class="sub-header">{subtitle}</div>'
        st.markdown(sub_html, unsafe_allow_html=True)


def create_info_box(message: str, box_type: str = "info"):
    """
    Create styled info box.
    
    Args:
        message: Message to display
        box_type: One of 'info', 'success', 'warning', 'error'
    """
    import streamlit as st
    
    box_html = f'<div class="{box_type}-box">{message}</div>'
    st.markdown(box_html, unsafe_allow_html=True)


def create_metric_card(label: str, value: str, unit: str = ""):
    """
    Create metric card HTML.
    
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
