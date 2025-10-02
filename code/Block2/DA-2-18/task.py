import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from src.tools.parser import get_parser
from src.tools.config_loader import load_config, Config


def generate_synthetic_data(opts: Config) -> pd.DataFrame:
    """Generate synthetic categorical data for chi-square test analysis.
    
    Parameters
    ----------
    opts : Config
        Configuration object containing data generation parameters
    
    Returns
    -------
    pd.DataFrame
        DataFrame with two categorical columns
    """

    # Синтетику мы генерируем сами
    try:
        np.random.seed(opts.data.random_seed)
        
        n_samples = opts.data.sample_size
        var1_cats = opts.variables.var1_categories
        var2_cats = opts.variables.var2_categories
        
        # Вероятности для первой переменной
        var1_probs = np.random.dirichlet(np.ones(len(var1_cats)))
        var1_probs = list(var1_probs / var1_probs.sum())
        
        var1 = np.random.choice(var1_cats, size=n_samples, p=var1_probs)

        # Динамическое создание зависимостей на основе индексов
        var2 = []
        for cat in var1:
            cat_index = var1_cats.index(cat)
            
            # Создаем вероятности, которые зависят от индекса категории
            # Это создает "диагональную" зависимость
            base_probs = np.ones(len(var2_cats))
            # Увеличиваем вероятность для категории с тем же индексом
            if cat_index < len(var2_cats):
                base_probs[cat_index] += 0.2

            probs = base_probs / base_probs.sum()
            var2.append(np.random.choice(var2_cats, p=probs))
            
        return pd.DataFrame({
            'Category1': var1,
            'Category2': var2
        })
        
    except Exception as e:
        print(f"Ошибка при генерации синтетики: {e}")
        raise

def perform_chi2_analysis(df: pd.DataFrame, opts: Config) -> dict | None:
    """Perform chi-square test of independence for two categorical variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing two categorical columns
    opts : Config
        Configuration object
    
    Returns
    -------
    dict | None
        Dictionary containing test results and statistics
    """

    try:
        contingency_table = pd.crosstab(df['Category1'], df['Category2'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print("Таблица сопряженности (Contingency Table):")
        print(contingency_table)
        print("\n" + "="*50 + "\n")
        
        # Тут идет интерпретация результатов
        alpha = 0.05
        if p_value < alpha:
            interpretation = "ЕСТЬ статистически значимая зависимость"
        else:
            interpretation = "НЕТ статистически значимой зависимости"
        
        results = {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'contingency_table': contingency_table,
            'expected_frequencies': expected,
            'interpretation': interpretation,
            'significance_level': alpha
        }
    except Exception as e:
        print(f"Ошибка при подсчете хи-квадрат: {e}")
        return None
    
    return results


def print_results(results: dict):
    """Print chi-square test results in a formatted way.
    
    Parameters
    ----------
    results : dict
        Dictionary containing test results and statistics
    """
    print("РЕЗУЛЬТАТЫ ТЕСТА ХИ-КВАДРАТ:")
    print("=" * 40)
    print(f"Хи-квадрат статистика: {results['chi2_statistic']:.4f}")
    print(f"p-value: {results['p_value']:.10f}")
    print(f"Степени свободы: {results['degrees_of_freedom']} (показывают, сколько ячеек в таблице можно заполнить 'произвольно', при известных итоговых суммах по строкам и столбцам)")
    print(f"Порог значимости: {results['significance_level']}")
    print("\nИНТЕРПРЕТАЦИЯ:")
    print(f"→ {results['interpretation']}")
    print(f"→ p-value {'<' if results['p_value'] < results['significance_level'] else '>='} {results['significance_level']}")


def save_results(results: dict, opts: Config):
    """Save analysis results to file.
    
    Parameters
    ----------
    results : dict
        Dictionary containing test results
    opts : Config
        Configuration object with output path
    """

    # Записываем результаты в текстовик, на всякий случай
    try:
        with open(opts.output.results_path, 'w', encoding='utf-8') as f:
            f.write("АНАЛИЗ ЗАВИСИМОСТИ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ\n")
            f.write("=" * 50 + "\n\n")
            f.write("Таблица сопряженности:\n")
            f.write(str(results['contingency_table']) + "\n\n")
            f.write("Результаты теста хи-квадрат:\n")
            f.write(f"Статистика хи-квадрат: {results['chi2_statistic']:.4f}\n")
            f.write(f"p-value: {results['p_value']:.10f}\n")
            f.write(f"Степени свободы: {results['degrees_of_freedom']}\n")
            f.write(f"Уровень значимости: {results['significance_level']}\n\n")
            f.write(f"ВЫВОД: {results['interpretation']}\n")
        print(f"\nРезультаты сохранены в: {opts.output.results_path}")
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")


def analyze_categorical_dependence(opts: Config):
    """Main function to perform categorical variables dependence analysis.
    
    Parameters
    ----------
    opts : Config
        Configuration object with all analysis parameters
    """
    print("Генерация синтетических данных...")
    df = generate_synthetic_data(opts)
    
    print("Первые 10 строк данных:")
    print(df.head(10))
    print("\n" + "="*50 + "\n")
    
    print("Распределение по переменным:")
    print(df['Category1'].value_counts())
    print("\n")
    print(df['Category2'].value_counts())
    print("\n" + "="*50 + "\n")
    
    results = perform_chi2_analysis(df, opts)
    
    if results is not None:
        print_results(results)
        save_results(results, opts)


def main():
    """Main entry point for the categorical analysis program."""
    # Парсим аргументы командной строки
    parser = get_parser()
    args = parser.parse_args()

    # Создаем отдельный объект класса Config 
    # (удобно для дальнейшей работы и большого коо-ва параметров)
    opts = load_config(args.config_path)

    analyze_categorical_dependence(opts)


if __name__ == "__main__":
    main()