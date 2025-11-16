# Polymer Production Scheduler V2

Modern, modular production scheduling optimizer using CP-SAT.

## Features

- 🎯 Multi-plant production optimization with shutdown management
- 📊 Real-time visualization with interactive Gantt charts
- 🔧 Flexible constraint management (min/max run days, transitions)
- 📈 Advanced inventory tracking and stockout prevention
- 🚀 Optimized CP-SAT solver with configurable parameters
- 📱 Responsive Streamlit UI with modern design

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/polymer-production-scheduler-v2.git
cd polymer-production-scheduler-v2

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

Visit http://localhost:8501 in your browser.

### Using Sample Template

1. Download the sample template from the app
2. Fill in your data following the format
3. Upload and run optimization

## Project Structure

```
src/
├── config/     # Configuration and settings
├── core/       # Optimization solver logic
├── data/       # Data loading and validation
├── ui/         # Streamlit UI components
├── models/     # Data models (Pydantic)
└── utils/      # Helper functions
```

## Excel File Format

Your Excel file should contain:

1. **Plant Sheet**: Production line capacities and shutdowns
2. **Inventory Sheet**: Grade configurations and constraints
3. **Demand Sheet**: Daily demand forecast by grade
4. **Transition Sheets** (optional): Allowed grade transitions per plant

See sample template for detailed format.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

## Documentation

- Architecture: Modular design with clear separation of concerns
- Solver: Google OR-Tools CP-SAT constraint programming
- UI: Streamlit with custom theming

## License

MIT License - See LICENSE file for details

## Support

For issues or questions, please open a GitHub issue.
