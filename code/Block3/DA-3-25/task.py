import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src.tools.parser import get_parser
from src.tools.config_loader import load_config, Config


def generate_synthetic_data(opts: Config) -> pd.DataFrame:
    """Generate synthetic time series data with various trends.
    
    Parameters
    ----------
    opts : Config
        Configuration object containing data generation parameters
    
    Returns
    -------
    pd.DataFrame
        DataFrame with time index and synthetic values
    """
    np.random.seed(opts.data.random_seed)
    n_samples = opts.data.sample_size
    
    # Создаем временной ряд с разными компонентами
    time_index = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    
    # Базовый тренд в зависимости от типа
    trend_type = getattr(opts.data, 'trend_type', 'sinusoidal')
    
    if trend_type == 'sinusoidal':
        trend = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 100)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
        noise = np.random.normal(0, 2, n_samples)
        values = trend + seasonal + noise + 50
        
    elif trend_type == 'linear':
        trend = 0.1 * np.arange(n_samples)
        seasonal = 3 * np.sin(2 * np.pi * np.arange(n_samples) / 50)
        noise = np.random.normal(0, 3, n_samples)
        values = trend + seasonal + noise + 30
        
    else:  # random
        values = np.cumsum(np.random.normal(0, 2, n_samples)) + 50
    
    # Добавляем выбросы для интереса
    outlier_indices = np.random.choice(n_samples, size=10, replace=False)
    values[outlier_indices] += np.random.normal(20, 10, 10)
    
    return pd.DataFrame({
        'value': values,
        'timestamp': time_index
    }).set_index('timestamp')


def calculate_rolling_range_count(df: pd.DataFrame, opts: Config) -> pd.DataFrame:
    """Calculate rolling count of values within [Q1, Q3] range for each window.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time series data
    opts : Config
        Configuration object with window parameters
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional rolling range count feature
    """
    window_size = opts.window.size
    min_periods = getattr(opts.window, 'min_periods', 1)
    
    def count_in_iqr(window):
        """Count values within [Q1, Q3] range for given window."""
        if len(window) < 2:
            return 0
        q1 = np.percentile(window, 25)
        q3 = np.percentile(window, 75)
        return np.sum((window >= q1) & (window <= q3))
    
    # Вычисляем скользящее количество значений в [Q1, Q3]
    df['rolling_range_count'] = (
        df['value']
        .rolling(window=window_size, min_periods=min_periods)
        .apply(count_in_iqr, raw=True)
    )
    
    # Дополнительные метрики для анализа
    df['rolling_mean'] = df['value'].rolling(window=window_size, min_periods=min_periods).mean()
    df['rolling_std'] = df['value'].rolling(window=window_size, min_periods=min_periods).std()
    df['rolling_q1'] = df['value'].rolling(window=window_size, min_periods=min_periods).quantile(0.25)
    df['rolling_q3'] = df['value'].rolling(window=window_size, min_periods=min_periods).quantile(0.75)
    
    return df


def plot_analysis_results(df: pd.DataFrame, opts: Config):
    """Create comprehensive visualization of rolling range analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with calculated features
    opts : Config
        Configuration object with output paths
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Анализ скользящего количества значений в диапазоне [Q1, Q3]', 
                 fontsize=16, fontweight='bold')
    
    # График 1: Исходные данные и скользящие квантили
    axes[0].plot(df.index, df['value'], label='Исходные данные', 
                 color='blue', alpha=0.7, linewidth=1)
    axes[0].plot(df.index, df['rolling_mean'], label='Скользящее среднее', 
                 color='red', linewidth=2)
    axes[0].fill_between(df.index, df['rolling_q1'], df['rolling_q3'], 
                        alpha=0.3, color='orange', label='Диапазон [Q1, Q3]')
    axes[0].set_ylabel('Значения')
    axes[0].set_title('Исходные данные и скользящие статистики')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График 2: Скользящее количество значений в [Q1, Q3]
    axes[1].plot(df.index, df['rolling_range_count'], 
                 color='green', linewidth=2, label='Количество в [Q1, Q3]')
    axes[1].axhline(y=opts.window.size * 0.5, color='red', linestyle='--', 
                   label='Ожидаемое (50% от окна)')
    axes[1].set_ylabel('Количество значений')
    axes[1].set_title('Скользящее количество значений в диапазоне [Q1, Q3]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # График 3: Процент значений в [Q1, Q3]
    df['rolling_range_percent'] = (df['rolling_range_count'] / opts.window.size) * 100
    axes[2].plot(df.index, df['rolling_range_percent'], 
                 color='purple', linewidth=2, label='Процент в [Q1, Q3]')
    axes[2].axhline(y=50, color='red', linestyle='--', 
                   label='Теоретический процент (50%)')
    axes[2].set_ylabel('Процент (%)')
    axes[2].set_xlabel('Время')
    axes[2].set_title('Процент значений в диапазоне [Q1, Q3]')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(opts.output.results_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Дополнительная статистика
    print("\n" + "="*60)
    print("СТАТИСТИЧЕСКИЙ АНАЛИЗ:")
    print("="*60)
    
    valid_counts = df['rolling_range_count'].dropna()
    if len(valid_counts) > 0:
        expected_count = opts.window.size * 0.5  # Теоретическое ожидание для нормального распределения
        
        print(f"Размер окна: {opts.window.size}")
        print(f"Теоретическое ожидание: {expected_count:.1f} значений (50%)")
        print(f"Фактическое среднее: {valid_counts.mean():.2f} значений")
        print(f"Фактический средний процент: {valid_counts.mean() / opts.window.size * 100:.1f}%")
        print(f"Стандартное отклонение: {valid_counts.std():.2f}")
        print(f"Минимальное количество: {valid_counts.min():.0f}")
        print(f"Максимальное количество: {valid_counts.max():.0f}")
        
        # Проверка отклонения от теоретического значения
        t_stat, p_value = stats.ttest_1samp(valid_counts, expected_count)
        print(f"\nТест на отклонение от теоретического значения (50%):")
        print(f"t-статистика: {t_stat:.3f}, p-value: {p_value:.3f}")
        
        if p_value < 0.05:
            print("→ Статистически значимое отклонение от 50% (p < 0.05)")
        else:
            print("→ Отклонение от 50% статистически не значимо")


def save_analysis_data(df: pd.DataFrame, opts: Config):
    """Save analysis results to CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with calculated features
    opts : Config
        Configuration object with output paths
    """
    try:
        # Сохраняем данные для дальнейшего анализа
        output_df = df.reset_index()
        output_df.to_csv(opts.output.data_path, index=False, encoding='utf-8')
        print(f"\nДанные сохранены в: {opts.output.data_path}")
    except Exception as e:
        print(f"Ошибка при сохранении данных: {e}")


def analyze_rolling_range(opts: Config):
    """Main function to perform rolling range count analysis.
    
    Parameters
    ----------
    opts : Config
        Configuration object with all analysis parameters
    """
    print("Генерация синтетических временных данных...")
    df = generate_synthetic_data(opts)
    
    print(f"Размер данных: {len(df)} записей")
    print(f"Тип тренда: {opts.data.trend_type}")
    print(f"Размер скользящего окна: {opts.window.size}")
    print("\n" + "="*60)
    
    print("Расчет скользящего количества значений в [Q1, Q3]...")
    df = calculate_rolling_range_count(df, opts)
    
    print("Построение графиков...")
    plot_analysis_results(df, opts)
    
    print("Сохранение результатов...")
    save_analysis_data(df, opts)


def main():
    """Main entry point for the rolling range analysis program."""
    parser = get_parser()
    args = parser.parse_args()
    opts = load_config(args.config_path)
    analyze_rolling_range(opts)


if __name__ == "__main__":
    main()