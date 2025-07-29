import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df


def prepare_features(df, etf):
    df[f"ret_1d_{etf}"] = np.log(df[f"Close_{etf}"] / df[f"Open_{etf}"])
    df['target'] = df[f"ret_1d_{etf}"].shift(-1)
    df = df.dropna(subset=['target'])
    # Features: all columns except target and any future-leaking columns
    X = df.drop(columns=['target'])
    y = df['target'].values
    return X.values, y


def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/master_df.csv', help='Path to master CSV')
    parser.add_argument('--etf',  default='SPY', help='ETF ticker for target')
    parser.add_argument('--out',  default='models/nn_model.h5', help='Output model file (.h5)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch',  type=int, default=32, help='Batch size')
    args = parser.parse_args()

    # Load and prepare
    df = load_data(args.data)
    X, y = prepare_features(df, args.etf)

    # Train/test split (80/20 time-based)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Ensure output dir
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    scaler_path = args.out.replace('.h5', '_scaler.pkl')
    pd.to_pickle(scaler, scaler_path)

    # Build and train model
    model = build_model(X_train.shape[1])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(args.out, save_best_only=True, monitor='val_loss')
    ]
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=2
    )

    # Evaluate on test set
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {mae:.6f}")

    # Save final model architecture and weights
    model.save(args.out)
    print(f"Model saved to {args.out}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == '__main__':
    main()