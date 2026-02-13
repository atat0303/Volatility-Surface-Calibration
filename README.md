# Volatility Surface Calibration - SVI Model

A Python project for calibrating volatility surfaces using the Stochastic Volatility Inspired (SVI) model on SPX options data.

## 📁 Project Structure

```
Volatility-Surface-Calibration/
├── .env                          # Environment configuration (not in git)
├── .env-dev                      # Development environment template
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── venv/                         # Virtual environment (not in git)
├── data/
│   ├── raw/                      # Raw SPX options data
│   │   ├── spx_eod_2023q1-cfph7w/
│   │   │   ├── spx_eod_202301.txt
│   │   │   ├── spx_eod_202302.txt
│   │   │   └── spx_eod_202303.txt
│   │   ├── spx_eod_2023q2-kdxt36/
│   │   ├── spx_eod_2023q3-w9b0jk/
│   │   └── spx_eod_2023q4-ai4uc9/
│   └── cleaned_data/             # Cleaned/processed data
│       └── temp_clean.csv
├── src/
│   └── data_cleaning.py          # Data cleaning and preprocessing script
└── visualizations/
    └── data_cleaning/            # Data quality visualizations
        ├── iv_smile.png
        ├── term_structure.png
        └── liquidity_moneyness.png
```

## Initial Setup

### 1. Clone the Repository
```bash
cd /path/to/your/projects
git clone <repository-url>
cd Volatility-Surface-Calibration
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
The project uses a `.env` file for configuration. Copy or edit the `.env` file:

```bash
# .env file contents:
DATA_DIR=data/raw/spx_eod_2023q1-cfph7w
DATA_FILE_NAME=spx_eod_202301.txt
CLEANED_DATA_DIR=data/cleaned_data
CLEANED_FILE_NAME=temp_clean.csv
VIZ_DIR=visualizations/data_cleaning
```

**To switch data files:** Simply edit the `.env` file to point to different data files (e.g., different months or quarters).

## Data Cleaning

### Purpose
The `data_cleaning.py` script:
- Loads raw SPX options data
- Cleans and filters options data (removes invalid spreads, negative prices, etc.)
- Normalizes implied volatilities
- Generates data quality reports
- Saves cleaned data and visualizations

### How to Run

**From repository root:**
```bash
python -m src.data_cleaning
```

**Current Working Directory:** Must be the repository root (`Volatility-Surface-Calibration/`)

### Outputs

1. **Cleaned Data CSV:** `data/cleaned_data/temp_clean.csv`
   - Processed options data ready for modeling

2. **Console Output:**
   - Dataset statistics (rows removed, percentage filtered)
   - Spread statistics
   - Implied volatility statistics
   - Arbitrage checks
   - Moneyness and maturity coverage

3. **Visualizations:** `visualizations/data_cleaning/`
   - `iv_smile.png` - Implied volatility smile (log-moneyness vs IV)
   - `term_structure.png` - Volatility term structure (maturity vs IV)
   - `liquidity_moneyness.png` - Liquidity analysis (spread vs moneyness)

## Dependencies

Core packages:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Plotting and visualization
- `python-dotenv` - Environment variable management

See [`requirements.txt`](requirements.txt) for full dependency list.

## 🔧 Configuration

### Environment Variables (`.env`)

| Variable | Description | Example |
|----------|-------------|---------|
| `DATA_DIR` | Directory containing raw data | `data/raw/spx_eod_2023q1-cfph7w` |
| `DATA_FILE_NAME` | Raw data filename | `spx_eod_202301.txt` |
| `CLEANED_DATA_DIR` | Output directory for cleaned data | `data/cleaned_data` |
| `CLEANED_FILE_NAME` | Cleaned data filename | `temp_clean.csv` |
| `VIZ_DIR` | Visualization output directory | `visualizations/data_cleaning` |

## 🎯 Usage Examples

### Process Different Data Files

**January 2023:**
```bash
# Edit .env
DATA_FILE_NAME=spx_eod_202301.txt

# Run
python -m src.data_cleaning
```

**February 2023:**
```bash
# Edit .env
DATA_FILE_NAME=spx_eod_202302.txt

# Run
python -m src.data_cleaning
```

## 🧪 Development Workflow

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Make code changes**

3. **Run data cleaning:**
   ```bash
   python -m src.data_cleaning
   ```

4. **Check outputs:**
   - Review console statistics
   - Inspect `data/cleaned_data/temp_clean.csv`
   - View plots in `visualizations/data_cleaning/`

5. **Commit changes:**
   ```bash
   git add .
   git commit -m "Description of changes"
   git push
   ```

## 📝 Notes

- **Working Directory:** Always run scripts from the repository root
- **Module Execution:** Use `python -m` for proper module imports
- **Virtual Environment:** Always activate `venv` before running scripts
- **Data Files:** Raw data files are not tracked in git (see `.gitignore`)
- **Configuration:** Use `.env` for switching between data files - never hardcode paths in scripts

## Future Work

- [ ] SVI model implementation
- [ ] Volatility surface calibration
- [ ] Model validation and backtesting
- [ ] Additional data sources integration

## License

[Add license information]

## Contributors

[Add contributor information]