"""
Utility functions used across the application.
"""

from datetime import date


def format_date(d: date) -> str:
    """
    Format date consistently.
    
    Args:
        d: Date to format
    
    Returns:
        Formatted date string (DD-Mon-YY)
    """
    return d.strftime('%d-%b-%y')


def parse_comma_separated(value: str) -> list:
    """
    Parse comma-separated string.
    
    Args:
        value: Comma-separated string
    
    Returns:
        List of trimmed values
    """
    if not value or str(value).strip() == '':
        return []
    return [x.strip() for x in str(value).split(',')]
```

---

## 📋 COMPLETE FILE STRUCTURE

Upload these files to your repository in this **exact structure**:
```
your-repo/
├── app.py                          ← Upload FILE 5 (minimal orchestration)
├── requirements.txt                ← Keep existing
├── README.md                       ← Keep existing
├── .gitignore                      ← Keep existing
├── .streamlit/
│   └── config.toml                 ← Keep existing
├── assets/
│   └── polymer_production_template.xlsx  ← Keep existing
└── src/
    ├── __init__.py                 ← Create empty file
    ├── core/
    │   ├── __init__.py            ← Create empty file
    │   └── solver.py              ← Upload FILE 4
    ├── data/
    │   ├── __init__.py            ← Create empty file
    │   └── loaders.py             ← Upload FILE 3
    ├── ui/
    │   ├── __init__.py            ← Create empty file
    │   ├── styles.py              ← Upload FILE 1
    │   ├── components.py          ← Upload FILE 2
    │   └── visualizations.py      ← Already created (artifact 1)
    └── utils/
        ├── __init__.py            ← Create empty file
        └── helpers.py             ← Upload FILE 6 (above)
