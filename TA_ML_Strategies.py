# /ta_ml_trade.py
import os
import json
import asyncio
import joblib
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from datetime import datetime, timedelta, timezone
import optuna
from dotenv import load_dotenv
import talib
from typing import Optional
from pydantic import ValidationError, SecretStr, BaseModel, model_validator, Field
# from lightgbm import LGBMClassifier
# from lightgbm import early_stopping
import lightgbm as lgb


from utils import (
    send_telegram_message, 
    remove_timezone,
    DataFetcher, 
    FetcherConfig,
    load_module_from_path
    )
# from model import train_model
from backtesting import Backtest, Strategy
from settings import bot_settings as bs

WAY = bs.WAY
COMMISSION = bs.COMMISSION
LIMIT = bs.LIMIT

N_TRIALS=200
OPTUNA_PERIOD_DAYS = 60

load_dotenv()
api_key = load_dotenv("api_key_bybit")
api_secret = load_dotenv("api_secret_bybit")
print(f"API Key: {api_key}")
print(f"API Secret: {api_secret}")


class TrainStrategy:
    def __init__(self, cfg: dict):
        load_dotenv()

        self.coin = self._get_config_value(cfg, "coin")
        self.interval = self._get_config_value(cfg, "interval")
        self.deposit = self._get_config_value(cfg, "deposit")
        self.leverage = self._get_config_value(cfg, "leverage")
        self.tp = self._get_config_value(cfg, "tp")
        self.sl = self._get_config_value(cfg, "sl")
        self.strategy = self._get_config_value(cfg, "strategy")

        self.folder = os.path.join(WAY, f"models/{self.coin}_{self.interval}_{self.strategy}")
        os.makedirs(self.folder, exist_ok=True)

        self.api_key = self._get_env_variable("api_key_bybit")
        self.api_secret = self._get_env_variable("api_secret_bybit")

        self._MyStrategy = self._create_strategy_class()

    def _get_config_value(self, cfg: dict, key: str):
        if key not in cfg:
            raise ValueError(f"[❌] В конфигурации отсутствует ключ: '{key}'")
        return cfg[key]

    def _get_env_variable(self, key: str):
        value = os.getenv(key)
        if value is None:
            raise EnvironmentError(f"[❌] Не найдена переменная окружения: '{key}'")
        return value

    def fetch_data(self):
        fetcher_config = FetcherConfig(
            api_key=self.api_key,
            api_secret=self.api_secret,
            period=timedelta(days=90),
            interval=self.interval,
            coin=self.coin,
        )
        fetcher = DataFetcher(config=fetcher_config)
        df = fetcher.fetch_coin_data()
        df.columns = df.columns.str.lower()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df.set_index("date", inplace=True)
        return df

    def train(self):
        df = self.fetch_data()
        # optuna_period_days = 20
        df_optuna = df[df.index >= df.index.max() - timedelta(days=OPTUNA_PERIOD_DAYS)].copy()

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self._objective(trial, df_optuna, self.deposit), n_trials=N_TRIALS, n_jobs=-1)
        best_params = study.best_params

        with open(os.path.join(self.folder, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)

        split = int(len(df) * 0.8)
        df_train, df_test = df.iloc[:split], df.iloc[split:]

        df_train = add_features(df_train, **best_params)
        df_train["Target"] = (df_train["Close"].shift(-1) > df_train["Close"]).astype(int)
        df_train.dropna(inplace=True)

        model = self._train_model(df_train)
        joblib.dump(model, os.path.join(self.folder, "model.pkl"))

        df_test = add_features(df_test, **best_params)
        # df_test["Target"] = (df_test["Close"].shift(-1) > df_test["Close"]).astype(int)

        # feature_columns = X_train.columns.tolist()
        X_test = df_test.copy() #[feature_columns]
        df_test.dropna(inplace=True)
        # Выбираем те же признаки, что и при обучении

        X_test = X_test.loc[:, ~X_test.columns.duplicated()]

        
        predicted_probabilities = model.predict(X_test)
        predicted_classes = np.argmax(predicted_probabilities, axis=1)

        signals = predicted_classes - 1
        signal_series = pd.Series(signals, index=X_test.index, name="Signal")

        df_test['Signal'] = pd.Series(signals, index=X_test.index)
        print(f"[INFO] Сгенерировано сигналов: Buy(1)={sum(signals==1)}, Sell(-1)={sum(signals==-1)}")

        bt = Backtest(df_test, self._MyStrategy, cash=self.deposit, commission=COMMISSION, exclusive_orders=True)
        stats = bt.run()
        stats.to_json(os.path.join(self.folder, "test_metrics_backtest.json"))

        try:
            bt.plot(filename=os.path.join(self.folder, "test_backtest_plot.html"), open_browser=False)
        except:
            print('Ошибка в графике')
        
        try:
            bt.plot(plot_equity=True, plot_drawdown=True, relative_equity=False, resample=False)
        except:
            print('График не обработан')
        print(stats[:27])

        main_stats_df = pd.DataFrame([stats])
        main_stats_df = main_stats_df.T.reset_index().rename(columns={'index': 'metric', 0: 'value'})
        main_stats_df['value'] = main_stats_df['value'].apply(remove_timezone)

        main_stats_df.head(28).to_excel(
            os.path.join(self.folder, "test_metrics_backtest.xlsx"), index=False
        )

        print(f"[✅] Обучена модель для {self.coin}_{self.interval}_{self.strategy}")

    def _train_model(self, df, num_classes=3):
        y_train = df['Target']
        X_train = df.drop(columns=['Target'])
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]

        train_data = lgb.Dataset(X_train, label=y_train)

        params = {
            'objective': 'multiclass',
            'num_class': num_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'verbose': -1,
            'random_state': 42
        }

        model = lgb.train(params, train_data)
        return model

    def _objective(self, trial, df, deposit):
        macd_fast = trial.suggest_int('macd_fast', 2, 50, step=2)
        macd_slow = trial.suggest_int('macd_slow', 10, 500, step=5)
        macd_signal = trial.suggest_int('macd_signal', 5, 100, step=5)
        sma_fast = trial.suggest_int('sma_fast', 5, 100, step=5)
        sma_slow = trial.suggest_int('sma_slow', 20, 100, step=2)
        rsi_period = trial.suggest_int('rsi_period', 2, 10, step=1)

        train_size = int(len(df) * 0.75)
        X_train = df[:train_size]

        df_train = add_features(X_train, macd_fast, macd_slow, macd_signal, sma_fast, sma_slow, rsi_period)
        df_train.dropna(inplace=True)

        bt = Backtest(df_train, self._MyStrategy, cash=deposit, commission=COMMISSION, exclusive_orders=True)
        stats = bt.run()

        if stats['# Trades'] < 50:
            return -10000
        return stats['Return [%]']

    def _create_strategy_class(self):
        outer_self = self
        class _MyStrategy(Strategy):
            def init(self):
                self.signal = self.I(lambda: self.data.Signal)

            def next(self):
                signal = self.data.df['Signal'].iloc[-1]
                current_price = self.data.Close[-1]

                if signal == 1:
                    self.buy(
                        sl=current_price * (1 - outer_self.sl),
                        tp=current_price * (1 + outer_self.tp)
                    )
                elif signal == -1:
                    self.sell(
                        sl=current_price * (1 + outer_self.sl),
                        tp=current_price * (1 - outer_self.tp)
                    )
        return _MyStrategy



'''_________________________ Создание признаков __________________________________'''

def add_features(df, 
                 macd_fast=12, 
                 macd_slow=26, 
                 macd_signal=9, 
                 sma_fast=10, 
                 sma_slow=50, 
                 rsi_period=14):
    
    df = df.copy()  # Создаем копию, чтобы не изменять исходный df
    df.columns = df.columns.str.capitalize()
    df = df[~df.index.duplicated(keep='last')]

    # MACD, SMA, RSI
    df['RSI'] = talib.RSI(df['Close'], timeperiod=rsi_period)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'], fastperiod=macd_fast, slowperiod=macd_slow,
                                                  signalperiod=macd_signal)
    df['SMA_Fast'] = talib.SMA(df['Close'], timeperiod=sma_fast)
    df['SMA_Slow'] = talib.SMA(df['Close'], timeperiod=sma_slow)

    # Сигналы
    df['Signal'] = np.where(
        (df['RSI'] < 30), 1,  # Покупка
        np.where((df['RSI'] > 70), -1, 0)  # Продажа / Ожидание
    )
    # df['Signal'] = np.where( (df['MACD'] > df['MACD_signal']) & (df['SMA_Fast'] > df['SMA_Slow']) & (df['RSI'] < 30), 1,  # Покупка
    #                         np.where((df['MACD'] < df['MACD_signal']) &  (df['SMA_Fast'] < df['SMA_Slow']) & (df['RSI'] > 70) , -1, 0))  # Продажа / Ожидание

    # Дополнительные RSI
    df['RSI_1'] = talib.RSI(df['Close'], timeperiod=5)
    df['RSI_2'] = talib.RSI(df['Close'], timeperiod=15)
    df['RSI_3'] = talib.RSI(df['Close'], timeperiod=50)

    # MACD с другими параметрами
    df['MACD_2'], df['MACD_signal_2'], df['MACD_slow_2'] = talib.MACD(df['Close'], fastperiod=15, slowperiod=60,
                                                                      signalperiod=3)

    # SMA с разными периодами
    df['SMA_1'] = talib.SMA(df['Close'], timeperiod=7)
    df['SMA_2'] = talib.SMA(df['Close'], timeperiod=15)
    df['SMA_3'] = talib.SMA(df['Close'], timeperiod=30)
    df['SMA_4'] = talib.SMA(df['Close'], timeperiod=100)

    # Дополнительные признаки
    df['d_min_max'] = df['High'] - df['Low']
    df['d_open_max'] = df['Open'] - df['High']
    df['d_open_min'] = df['Open'] - df['Low']

    # Уровни поддержки и сопротивления
    periods = [14, 20, 50, 100, 200]
    df = calculate_levels(df, periods=periods)

    # Дата и временные признаки
    try:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index('Date', inplace=True)
    except:
        pass

    df['weekday_number'] = df.index.weekday
    df['week_number'] = df.index.isocalendar().week

    # Динамика 1 час
    df['D_1T'] = df['Close'] - df['Open']
    df['D_1T'] = df['D_1T'].shift(1)

    # Импульс
    df['Im'] = 0
    limit_dynamics = np.percentile(df['D_1T'].dropna(), 30)
    df.loc[df['D_1T'] < limit_dynamics, 'Im'] = 1

    # Дополнительные технические индикаторы
    df['adx'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=9)
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=9)
    df['atr_norm'] = df['atr'] / df['Close']

    df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['Close'], timeperiod=7)
    df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']

    df['ema20'] = talib.EMA(df['Close'], timeperiod=20)
    df['ema50'] = talib.EMA(df['Close'], timeperiod=50)
    df['ema200'] = talib.EMA(df['Close'], timeperiod=200)

    df['macd_3'], df['macd_signal_3'], df['macd_hist_3'] = talib.MACD(df['Close'], fastperiod=15, slowperiod=20,
                                                                      signalperiod=9)

    df['sma_fast_2'] = talib.SMA(df['Close'], timeperiod=9)
    df['sma_slow_2'] = talib.SMA(df['Close'], timeperiod=30)
    df['tema'] = talib.TEMA(df['Close'], timeperiod=12)

    # Лаги
    df = create_lagged_features(df, 'tema', [2, 30])
    df = create_lagged_features(df, 'RSI_1', [2, 30])
    df = create_lagged_features(df, 'D_1T', [2, 30])
    df = create_lagged_features(df, 'Im', [2, 30])

    return df


def calculate_levels(data, periods=[14]):
    for period in periods:
        high_col = f'high_max_{period}'
        low_col = f'low_min_{period}'

        data.loc[:, high_col] = data['High'].rolling(window=period, min_periods=1).max()
        data.loc[:, low_col] = data['Low'].rolling(window=period, min_periods=1).min()

    return data


def create_lagged_features(data, column, lag_range):
    """
    Создает лагированные признаки для указанного столбца.
    """
    if column not in data.columns:
        raise ValueError(f"Колонка '{column}' отсутствует в DataFrame")

    data = data.copy()  # Избегаем предупреждений о копии данных
    data = data[~data.index.duplicated(keep='last')]

    start, end = lag_range
    for i in range(start, end + 1):
        lag_col = f"{column}_lag_{i}h"
        try:
            data.loc[:, lag_col] = data[column].shift(periods=i, freq='h')
        except Exception as e:
            data.loc[:, lag_col] = 0
            print(f'Ошибка при создании лага {i}: {e}')

    return data


'''_________________________ Торговля __________________________________'''

class StrategyRunner:
    def __init__(self, cfg: dict):
        load_dotenv()
        self.cfg = cfg
        self.coin = cfg["coin"]
        self.interval_min = cfg["interval"]
        self.interval = self.interval_min * 60
        self.strategy = cfg["strategy"]
        self.name = f"{self.coin}_{self.interval_min}_{self.strategy}"

        self.api_key = os.getenv("api_key_bybit")
        self.api_secret = os.getenv("api_secret_bybit")

        self.session = HTTP(api_key=self.api_key, api_secret=self.api_secret)

        self.is_active = False
        self.last_signal = 0

        self.model_folder = os.path.join(WAY, "models", self.name)

        self.model = None
        self.params = None
        self.df = pd.DataFrame()
        self.fetcher_config = None
        self.data_fetcher = None

    async def _initialize(self) -> bool:
        params_path = os.path.join(self.model_folder, "best_params.json")
        with open(params_path, 'r') as f:
            self.params = json.load(f)

        model_path = os.path.join(self.model_folder, "model.pkl")
        self.model = joblib.load(model_path)

        fetcher_params = {
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            'period': timedelta(days=self.cfg.get('fetch_period_days', 10)),
            'interval': self.interval_min,
            'coin': self.coin,
        }
        if 'limit' in self.cfg:
            fetcher_params['limit'] = self.cfg['limit']
        if 'stock' in self.cfg:
            fetcher_params['stock'] = self.cfg['stock']
        if 'db_path' in self.cfg:
            fetcher_params['db_path'] = self.cfg['db_path']

        self.fetcher_config = FetcherConfig(**fetcher_params)
        self.data_fetcher = DataFetcher(config=self.fetcher_config)

        loop = asyncio.get_running_loop()
        updated_df = await loop.run_in_executor(None, self.data_fetcher.fetch_coin_data)
        if updated_df is not None and not updated_df.empty:
            updated_df.columns = updated_df.columns.str.lower()
            if 'date' in updated_df.columns:
                updated_df = updated_df.sort_values(by='date').reset_index(drop=True)
            self.df = updated_df

        return True

    async def wait_for_candle(self):
        now = datetime.now(timezone.utc)
        current_interval_start = now.replace(second=0, microsecond=0)
        seconds_past_interval_start = (now - current_interval_start.replace(minute=(now.minute // self.interval_min) * self.interval_min)).total_seconds()
        sleep_secs = self.interval - seconds_past_interval_start
        if sleep_secs <= 0:
            sleep_secs += self.interval
        await asyncio.sleep(sleep_secs)

    def generate_signal(self) -> int:
        print('Генерируем сигнал.')
        if self.df is None or self.df.empty:
            return 0
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in self.df.columns for col in required_cols):
            return 0

        df_copy = self.df.copy()
        df_copy = df_copy[~df_copy.index.duplicated(keep='last')].sort_values('Date')
        df_copy = add_features(df_copy, **self.params)

        df_model = df_copy.copy()
        df_model['Signal'] = df_model['Signal'] + 1
        df_model.columns = df_model.columns.str.capitalize()
        df_model = df_model.dropna()

        try:
            predictions = self.model.predict(df_model)
            predicted_signal = np.argmax(predictions, axis=1) - 1
            final_signal = int(predicted_signal[-1])
            confirmed_signal = int(df_copy['Signal'].iloc[-1])
            signal_date = df_copy['Date'].iloc[-1] if 'Date' in df_copy.columns else df_copy.index[-1]
            print(f"[🧠] {self.coin} | Time: {signal_date} | Signal: {confirmed_signal} | Predicted: {final_signal}")
            return final_signal if final_signal == confirmed_signal else 0
        except Exception as e:
            print(f"Ошибка при прогнозе: {e}")
            return 0

    def calc_qty(self, price: float) -> float:
        if price is None or price <= 0:
            return 0.0

        deposit = float(self.cfg['deposit'])
        leverage = float(self.cfg['leverage'])

        if price > 10000:
            round_value = 3
        elif price > 1000:
            round_value = 2
        elif price > 100:
            round_value = 1
        else:
            round_value = 0

        val = (deposit * leverage) / price
        qty = round(val, round_value)
        return qty if qty > 0 else 0.0

    def close_position(self):
        response = self.session.get_positions(category="linear", symbol=self.coin)
        if "result" in response and "list" in response["result"]:
            for pos in response["result"]["list"]:
                if float(pos["size"]) > 0:
                    side = "Sell" if pos["side"] == "Buy" else "Buy"
                    idx = 1 if pos["side"] == "Buy" else 2
                    self.session.place_order(
                        category="linear",
                        positionIdx=idx,
                        takeProfit=0,
                        stopLoss=0,
                        symbol=self.coin,
                        side=side,
                        orderType="Market",
                        qty=pos["size"],
                        timeInForce="GTC",
                        reduceOnly=True,
                        leverage=self.cfg['leverage']
                    )

    def cancel_all_orders(self):
        self.session.cancel_all_orders(
            category="linear",
            settleCoin="USDT",
            symbol=self.coin
        )

    async def _close_existing_position(self, reason: str):
        if not self.is_active:
            return

        print(f"[🔁] Закрытие позиции {self.coin} из-за: {reason}")
        self.close_position()
        self.cancel_all_orders()
        self.is_active = False
        self.last_signal = 0

    async def _place_new_order(self, signal: int, price: float):
        if signal == 0:
            return False

        side = 'Buy' if signal > 0 else 'Sell'
        tp_perc = self.cfg.get('tp', 0.01)
        sl_perc = self.cfg.get('sl', 0.005)
        leverage = float(self.cfg.get('leverage', 10))
        price_precision = self.cfg.get('price_precision', 2)

        qty = self.calc_qty(price)
        if qty <= 0:
            return False

        if signal > 0:
            tp_price = price * (1 + tp_perc)
            sl_price = price * (1 - sl_perc)
        else:
            tp_price = price * (1 - tp_perc)
            sl_price = price * (1 + sl_perc)

        tp_price = round(tp_price, price_precision)
        sl_price = round(sl_price, price_precision)
        order_price = round(price, price_precision)

        response = self.session.place_order(
            category="linear",
            symbol=self.coin,
            side=side,
            orderType='Limit',
            qty=str(qty),
            price=order_price,
            takeProfit=tp_price,
            stopLoss=sl_price,
            timeInForce="GTC",
            positionIdx=1 if signal > 0 else 2
        )

        if response['retCode'] == 0 and 'orderId' in response['result']:
            print(f"[🛒] Ордер размещен: {side} {self.coin} @ {order_price} | TP={tp_price}, SL={sl_price}, QTY={qty}")
            self.is_active = True
            self.last_signal = signal
            return True

        return False

    async def run(self):
        if not await self._initialize():
            return

        while True:
            await self.wait_for_candle()
            if not self.data_fetcher:
                continue

            loop = asyncio.get_running_loop()
            updated_df = await loop.run_in_executor(None, self.data_fetcher.fetch_coin_data)
            if updated_df is not None and not updated_df.empty:
                updated_df.columns = updated_df.columns.str.capitalize()
                if 'Date' in updated_df.columns:
                    updated_df = updated_df.sort_values(by='Date').reset_index(drop=True)
                self.df = updated_df

            if self.df is None or self.df.empty or 'Close' not in self.df.columns:
                continue

            last_price = self.df['Close'].iloc[-1]
            if not isinstance(last_price, (int, float)) or last_price <= 0:
                continue

            signal = self.generate_signal()
            if self.is_active:
                if signal == 0:
                    await self._close_existing_position("Сигнал 0 (выход)")
                elif signal == -self.last_signal:
                    await self._close_existing_position(f"Сигнал {signal} (переворот)")
                    await asyncio.sleep(1)
                    await self._place_new_order(signal, last_price)
            else:
                if signal != 0:
                    await self._place_new_order(signal, last_price)