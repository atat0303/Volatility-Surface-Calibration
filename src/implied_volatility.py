"""
Implied Volatility Calculator using Secant Method
Author: Daksh Kumar
Description: Calculate implied volatilities from cleaned market data using 
             Black-Scholes model with Secant root-finding method.
             Uses log-moneyness as convention.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Constants
# For 2023 data: Fed Funds rate was ~4-5% (using 4.5% average)
# SPX dividend yield ~1.5-2% (using 1.7% typical)
RISK_FREE_RATE = 0.045  # 4.5% - approximate 2023 risk-free rate
DIVIDEND_YIELD = 0.017  # 1.7% - SPX continuous dividend yield
MAX_ITERATIONS = 100
TOLERANCE = 1e-6
INITIAL_GUESS_1 = 0.2  # 20% vol
INITIAL_GUESS_2 = 0.3  # 30% vol

# Configuration from .env
CLEANED_DATA_DIR = os.getenv('CLEANED_DATA_DIR', 'data/cleaned_data')
CLEANED_FILE_NAME = os.getenv('CLEANED_FILE_NAME', 'temp_clean.csv')
IV_OUTPUT_DIR = 'data/implied_volatility'
IV_VIZ_DIR = 'visualizations/implied_volatility'

# Create output directories
os.makedirs(IV_OUTPUT_DIR, exist_ok=True)
os.makedirs(IV_VIZ_DIR, exist_ok=True)


def black_scholes_call(S, K, T, r, sigma, q=DIVIDEND_YIELD):
    """
    Calculate Black-Scholes call option price with dividend yield.
    
    Parameters:
    -----------
    S : float - Spot price
    K : float - Strike price
    T : float - Time to maturity (years)
    r : float - Risk-free rate
    sigma : float - Volatility
    q : float - Continuous dividend yield
    
    Returns:
    --------
    float - Call option price
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S, K, T, r, sigma, q=DIVIDEND_YIELD):
    """
    Calculate Black-Scholes put option price with dividend yield.
    
    Parameters:
    -----------
    S : float - Spot price
    K : float - Strike price
    T : float - Time to maturity (years)
    r : float - Risk-free rate
    sigma : float - Volatility
    q : float - Continuous dividend yield
    
    Returns:
    --------
    float - Put option price
    """
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price


def vega(S, K, T, r, sigma, q=DIVIDEND_YIELD):
    """
    Calculate vega (derivative of option price w.r.t. volatility).
    
    Parameters:
    -----------
    S : float - Spot price
    K : float - Strike price
    T : float - Time to maturity (years)
    r : float - Risk-free rate
    sigma : float - Volatility
    q : float - Continuous dividend yield
    
    Returns:
    --------
    float - Vega
    """
    if T <= 0 or sigma <= 0:
        return 0
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega_val = S * norm.pdf(d1) * np.sqrt(T)
    return vega_val


def implied_vol_secant(market_price, S, K, T, r, option_type='call', 
                       sigma0=INITIAL_GUESS_1, sigma1=INITIAL_GUESS_2,
                       max_iter=MAX_ITERATIONS, tol=TOLERANCE):
    """
    Calculate implied volatility using Secant method.
    
    Parameters:
    -----------
    market_price : float - Observed market price
    S : float - Spot price
    K : float - Strike price
    T : float - Time to maturity (years)
    r : float - Risk-free rate
    option_type : str - 'call' or 'put'
    sigma0 : float - First initial guess
    sigma1 : float - Second initial guess
    max_iter : int - Maximum iterations
    tol : float - Convergence tolerance
    
    Returns:
    --------
    float - Implied volatility (or np.nan if failed)
    """
    # Boundary checks
    if T <= 0:
        return np.nan

    # No-arbitrage bounds (with discounting and continuous dividend yield q)
    q = DIVIDEND_YIELD
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if option_type == 'call':
        lower_bound = max(S * disc_q - K * disc_r, 0.0)
        upper_bound = S * disc_q
    else:
        lower_bound = max(K * disc_r - S * disc_q, 0.0)
        upper_bound = K * disc_r

    # Fail fast on clearly invalid market prices
    if market_price < (lower_bound - 1e-12) or market_price > (upper_bound + 1e-12):
        return np.nan
    
    # Select pricing function
    bs_func = black_scholes_call if option_type == 'call' else black_scholes_put
    
    # Secant method
    for i in range(max_iter):
        # Calculate function values
        f0 = bs_func(S, K, T, r, sigma0) - market_price
        f1 = bs_func(S, K, T, r, sigma1) - market_price
        
        # Check convergence
        if abs(f1) < tol:
            return sigma1
        
        # Avoid division by zero
        if abs(f1 - f0) < 1e-10:
            return np.nan
        
        # Secant update
        sigma_new = sigma1 - f1 * (sigma1 - sigma0) / (f1 - f0)
        
        # Ensure positive volatility
        sigma_new = max(sigma_new, 0.001)
        
        # Check convergence on sigma
        if abs(sigma_new - sigma1) < tol:
            return sigma_new
        
        # Update for next iteration
        sigma0 = sigma1
        sigma1 = sigma_new
    
    # Failed to converge
    return np.nan


def calculate_implied_volatilities(df, r=RISK_FREE_RATE):
    """
    Calculate implied volatilities for all options in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame - Cleaned options data
    r : float - Risk-free rate
    
    Returns:
    --------
    pd.DataFrame - DataFrame with calculated IVs
    """
    df = df.copy()
    
    # Initialize columns for calculated IVs
    df['C_IV_CALC'] = np.nan
    df['P_IV_CALC'] = np.nan
    df['C_IV_ERROR'] = np.nan
    df['P_IV_ERROR'] = np.nan
    
    print("Calculating implied volatilities using Secant method...")
    print(f"Total options: {len(df)}")
    
    # Calculate for calls
    for idx, row in df.iterrows():
        # Call IV
        c_iv = implied_vol_secant(
            market_price=row['C_MID'],
            S=row['S'],
            K=row['K'],
            T=row['T'],
            r=r,
            option_type='call'
        )
        df.at[idx, 'C_IV_CALC'] = c_iv
        
        # Put IV
        p_iv = implied_vol_secant(
            market_price=row['P_MID'],
            S=row['S'],
            K=row['K'],
            T=row['T'],
            r=r,
            option_type='put'
        )
        df.at[idx, 'P_IV_CALC'] = p_iv
    
    # Calculate errors vs market IVs
    df['C_IV_ERROR'] = df['C_IV_CALC'] - df['C_IV_MID']
    df['P_IV_ERROR'] = df['P_IV_CALC'] - df['P_IV_MID']
    
    # Calculate success rate
    c_success = df['C_IV_CALC'].notna().sum()
    p_success = df['P_IV_CALC'].notna().sum()
    
    print(f"\nCall IV Success Rate: {c_success}/{len(df)} ({100*c_success/len(df):.2f}%)")
    print(f"Put IV Success Rate: {p_success}/{len(df)} ({100*p_success/len(df):.2f}%)")
    
    return df


def generate_iv_report(df):
    """
    Generate statistical report on calculated implied volatilities.
    
    Parameters:
    -----------
    df : pd.DataFrame - DataFrame with calculated IVs
    
    Returns:
    --------
    dict - Report statistics
    """
    report = {}

    # Guard: empty dataframe
    if len(df) == 0:
        return {
            'total_options': 0,
            'call_success': 0,
            'put_success': 0,
            'call_success_rate': 0.0,
            'put_success_rate': 0.0,
            'call_iv_mean': np.nan,
            'call_iv_std': np.nan,
            'put_iv_mean': np.nan,
            'put_iv_std': np.nan,
            'call_error_mean': np.nan,
            'call_error_std': np.nan,
            'call_error_mae': np.nan,
            'put_error_mean': np.nan,
            'put_error_std': np.nan,
            'put_error_mae': np.nan,
            'call_iv_corr': np.nan,
            'put_iv_corr': np.nan
        }

    # Filter successful calculations
    df_call = df[df['C_IV_CALC'].notna()]
    df_put = df[df['P_IV_CALC'].notna()]
    
    # Basic statistics
    report['total_options'] = len(df)
    report['call_success'] = len(df_call)
    report['put_success'] = len(df_put)
    report['call_success_rate'] = 100 * len(df_call) / len(df)
    report['put_success_rate'] = 100 * len(df_put) / len(df)
    
    # IV statistics
    report['call_iv_mean'] = df_call['C_IV_CALC'].mean()
    report['call_iv_std'] = df_call['C_IV_CALC'].std()
    report['put_iv_mean'] = df_put['P_IV_CALC'].mean()
    report['put_iv_std'] = df_put['P_IV_CALC'].std()
    
    # Error statistics
    report['call_error_mean'] = df_call['C_IV_ERROR'].mean()
    report['call_error_std'] = df_call['C_IV_ERROR'].std()
    report['call_error_mae'] = df_call['C_IV_ERROR'].abs().mean()
    report['put_error_mean'] = df_put['P_IV_ERROR'].mean()
    report['put_error_std'] = df_put['P_IV_ERROR'].std()
    report['put_error_mae'] = df_put['P_IV_ERROR'].abs().mean()
    
    # Correlation with market IVs
    report['call_iv_corr'] = df_call[['C_IV_MID', 'C_IV_CALC']].corr().iloc[0, 1]
    report['put_iv_corr'] = df_put[['P_IV_MID', 'P_IV_CALC']].corr().iloc[0, 1]
    
    return report


def plot_iv_comparison(df):
    """
    Generate visualization comparing calculated vs market IVs.
    
    Parameters:
    -----------
    df : pd.DataFrame - DataFrame with calculated IVs
    """
    df_valid = df[(df['C_IV_CALC'].notna()) & (df['P_IV_CALC'].notna())].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Implied Volatility Analysis - Secant Method', fontsize=16, y=1.00)
    
    # 1. Call IV: Calculated vs Market
    x_call = df_valid['C_IV_MID']
    y_call = df_valid['C_IV_CALC']
    axes[0, 0].scatter(x_call, y_call, alpha=0.5, s=10, c='blue')
    # Identity line spanning the range of call IVs
    call_min = np.nanmin([x_call.min(), y_call.min()])
    call_max = np.nanmax([x_call.max(), y_call.max()])
    axes[0, 0].plot([call_min, call_max], [call_min, call_max], 'r--', lw=2)
    axes[0, 0].set_xlabel('Market Call IV')
    axes[0, 0].set_ylabel('Calculated Call IV')
    axes[0, 0].set_title('Call IV: Calculated vs Market')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Put IV: Calculated vs Market
    x_put = df_valid['P_IV_MID']
    y_put = df_valid['P_IV_CALC']
    axes[0, 1].scatter(x_put, y_put, alpha=0.5, s=10, c='green')
    # Identity line spanning the range of put IVs
    put_min = np.nanmin([x_put.min(), y_put.min()])
    put_max = np.nanmax([x_put.max(), y_put.max()])
    axes[0, 1].plot([put_min, put_max], [put_min, put_max], 'r--', lw=2)
    axes[0, 1].set_xlabel('Market Put IV')
    axes[0, 1].set_ylabel('Calculated Put IV')
    axes[0, 1].set_title('Put IV: Calculated vs Market')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Call IV Error Distribution
    axes[0, 2].hist(df_valid['C_IV_ERROR'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 2].axvline(0, color='r', linestyle='--', lw=2)
    axes[0, 2].set_xlabel('IV Error (Calc - Market)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Call IV Error Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Put IV Error Distribution
    axes[1, 0].hist(df_valid['P_IV_ERROR'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].axvline(0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('IV Error (Calc - Market)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Put IV Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Call IV vs Log-Moneyness (Convention)
    axes[1, 1].scatter(df_valid['Y'], df_valid['C_IV_CALC'], 
                       alpha=0.4, s=10, c='blue', label='Calculated')
    axes[1, 1].scatter(df_valid['Y'], df_valid['C_IV_MID'], 
                       alpha=0.4, s=10, c='red', label='Market')
    axes[1, 1].set_xlabel('Log-Moneyness ln(K/S)')
    axes[1, 1].set_ylabel('Implied Volatility')
    axes[1, 1].set_title('Call IV Smile (Log-Moneyness Convention)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Put IV vs Log-Moneyness (Convention)
    axes[1, 2].scatter(df_valid['Y'], df_valid['P_IV_CALC'], 
                       alpha=0.4, s=10, c='green', label='Calculated')
    axes[1, 2].scatter(df_valid['Y'], df_valid['P_IV_MID'], 
                       alpha=0.4, s=10, c='red', label='Market')
    axes[1, 2].set_xlabel('Log-Moneyness ln(K/S)')
    axes[1, 2].set_ylabel('Implied Volatility')
    axes[1, 2].set_title('Put IV Smile (Log-Moneyness Convention)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IV_VIZ_DIR, 'iv_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {IV_VIZ_DIR}/iv_comparison.png")
    plt.close()


def plot_iv_surface_3d(df):
    """
    Generate 3D surface plot of implied volatility.
    
    Parameters:
    -----------
    df : pd.DataFrame - DataFrame with calculated IVs
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    df_valid = df[df['C_IV_CALC'].notna()].copy()
    
    fig = plt.figure(figsize=(16, 6))
    
    # Call IV Surface
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(df_valid['Y'], df_valid['T'], df_valid['C_IV_CALC'],
                          c=df_valid['C_IV_CALC'], cmap='viridis', s=5)
    ax1.set_xlabel('Log-Moneyness ln(K/S)')
    ax1.set_ylabel('Time to Maturity (Years)')
    ax1.set_zlabel('Implied Volatility')
    ax1.set_title('Call Implied Volatility Surface')
    plt.colorbar(scatter1, ax=ax1, pad=0.1)
    
    # Put IV Surface
    df_valid_put = df[df['P_IV_CALC'].notna()].copy()
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(df_valid_put['Y'], df_valid_put['T'], df_valid_put['P_IV_CALC'],
                          c=df_valid_put['P_IV_CALC'], cmap='plasma', s=5)
    ax2.set_xlabel('Log-Moneyness ln(K/S)')
    ax2.set_ylabel('Time to Maturity (Years)')
    ax2.set_zlabel('Implied Volatility')
    ax2.set_title('Put Implied Volatility Surface')
    plt.colorbar(scatter2, ax=ax2, pad=0.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IV_VIZ_DIR, 'iv_surface_3d.png'), dpi=150, bbox_inches='tight')
    print(f"3D Surface saved: {IV_VIZ_DIR}/iv_surface_3d.png")
    plt.close()


def main():
    """
    Main execution function.
    """
    print("="*80)
    print("IMPLIED VOLATILITY CALCULATOR - SECANT METHOD")
    print("Author: Daksh Kumar")
    print("Convention: Log-Moneyness ln(K/S)")
    print(f"Risk-Free Rate: {RISK_FREE_RATE*100:.2f}%")
    print(f"Dividend Yield: {DIVIDEND_YIELD*100:.2f}%")
    print("="*80)
    
    # Load cleaned data
    input_file = os.path.join(CLEANED_DATA_DIR, CLEANED_FILE_NAME)
    print(f"\nLoading cleaned data from: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} cleaned options records")
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        print("Please run data cleaning first: python -m src.data_cleaning")
        return
    
    # Calculate implied volatilities
    print("\n" + "="*80)
    df_with_iv = calculate_implied_volatilities(df, r=RISK_FREE_RATE)
    
    # Generate report
    print("\n" + "="*80)
    print("GENERATING STATISTICS REPORT")
    print("="*80)
    report = generate_iv_report(df_with_iv)
    
    print("\n===== IMPLIED VOLATILITY CALCULATION SUMMARY =====")
    print(f"Total Options:              {report['total_options']}")
    print(f"Call Success Rate:          {report['call_success_rate']:.2f}%")
    print(f"Put Success Rate:           {report['put_success_rate']:.2f}%")
    print(f"\nCall IV Statistics:")
    print(f"  Mean:                     {report['call_iv_mean']:.4f}")
    print(f"  Std Dev:                  {report['call_iv_std']:.4f}")
    print(f"  Correlation with Market:  {report['call_iv_corr']:.4f}")
    print(f"\nPut IV Statistics:")
    print(f"  Mean:                     {report['put_iv_mean']:.4f}")
    print(f"  Std Dev:                  {report['put_iv_std']:.4f}")
    print(f"  Correlation with Market:  {report['put_iv_corr']:.4f}")
    print(f"\nCall Error Statistics:")
    print(f"  Mean Error:               {report['call_error_mean']:.6f}")
    print(f"  Std Dev:                  {report['call_error_std']:.6f}")
    print(f"  MAE:                      {report['call_error_mae']:.6f}")
    print(f"\nPut Error Statistics:")
    print(f"  Mean Error:               {report['put_error_mean']:.6f}")
    print(f"  Std Dev:                  {report['put_error_std']:.6f}")
    print(f"  MAE:                      {report['put_error_mae']:.6f}")
    
    # Save results
    output_file = os.path.join(IV_OUTPUT_DIR, 'implied_volatility_secant.csv')
    df_with_iv.to_csv(output_file, index=False)
    print(f"\n✓ Implied volatility data saved: {output_file}")
    
    # Save report
    report_file = os.path.join(IV_OUTPUT_DIR, 'iv_calculation_report.txt')
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("IMPLIED VOLATILITY CALCULATION REPORT - SECANT METHOD\n")
        f.write("="*80 + "\n\n")
        for key, value in report.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ Report saved: {report_file}")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    plot_iv_comparison(df_with_iv)
    plot_iv_surface_3d(df_with_iv)
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - Data: {output_file}")
    print(f"  - Report: {report_file}")
    print(f"  - Visualizations: {IV_VIZ_DIR}/")


if __name__ == "__main__":
    main()
