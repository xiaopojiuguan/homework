"""手写 ANN 分类器 - 课内算法（前向传播 + 反向传播，从零实现）

实现一个全连接神经网络（MLP），包含：
- 前向传播（Forward Propagation）
- 反向传播（Backpropagation）
- 交叉熵损失 + Softmax 输出
- Mini-batch 梯度下降
- ReLU 隐藏层激活 + Softmax 输出层
- 可选 Dropout 正则化
"""
import numpy as np
from .base import BaseClassifier, AlgorithmFactory


class HandWrittenANN(BaseClassifier):
    """从零实现的多层感知机（MLP）"""

    def __init__(self, hidden_layers=None, learning_rate=0.01, epochs=100,
                 batch_size=64, dropout_rate=0.0, **kwargs):
        super().__init__("Hand-written ANN")
        self.hidden_layers = hidden_layers or [128, 64]
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.params = {}
        self.classes_ = None
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.input_dim = None
        self.output_dim = None

    def _init_weights(self, input_dim, output_dim):
        """He 初始化权重"""
        layers = [input_dim] + self.hidden_layers + [output_dim]
        for i in range(len(layers) - 1):
            self.params[f"W{i+1}"] = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            self.params[f"b{i+1}"] = np.zeros((1, layers[i+1]))

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(float)

    def _softmax(self, z):
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _forward(self, X, training=True):
        """前向传播"""
        cache = {"A0": X}
        n_layers = len(self.hidden_layers) + 1
        dropout_masks = {}

        A = X
        for i in range(n_layers - 1):
            Z = np.dot(A, self.params[f"W{i+1}"]) + self.params[f"b{i+1}"]
            A = self._relu(Z)
            cache[f"Z{i+1}"] = Z
            cache[f"A{i+1}"] = A

            if training and self.dropout_rate > 0:
                mask = (np.random.rand(*A.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
                A = A * mask
                dropout_masks[f"dropout{i+1}"] = mask

        # 输出层
        Z_out = np.dot(A, self.params[f"W{n_layers}"]) + self.params[f"b{n_layers}"]
        A_out = self._softmax(Z_out)
        cache[f"Z{n_layers}"] = Z_out
        cache[f"A{n_layers}"] = A_out
        cache["dropout_masks"] = dropout_masks

        return A_out, cache

    def _cross_entropy_loss(self, y_pred, y_true_onehot):
        """交叉熵损失"""
        eps = 1e-9
        loss = -np.sum(y_true_onehot * np.log(y_pred + eps)) / y_pred.shape[0]
        return loss

    def _backward(self, cache, y_true_onehot):
        """反向传播"""
        grads = {}
        n_layers = len(self.hidden_layers) + 1
        m = y_true_onehot.shape[0]

        # 输出层梯度: dL/dZ = (y_pred - y_true) / m
        dZ = (cache[f"A{n_layers}"] - y_true_onehot) / m

        for i in range(n_layers, 0, -1):
            # dW = A_prev^T @ dZ, db = sum(dZ)
            A_prev = cache[f"A{i-1}"]
            if i > 1 and self.dropout_rate > 0 and f"dropout{i-1}" in cache["dropout_masks"]:
                A_prev = A_prev * cache["dropout_masks"][f"dropout{i-1}"]

            grads[f"dW{i}"] = np.dot(A_prev.T, dZ)
            grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True)

            if i > 1:
                dA = np.dot(dZ, self.params[f"W{i}"].T)
                dZ = dA * self._relu_derivative(cache[f"Z{i-1}"])

        return grads

    def _update_params(self, grads):
        """梯度下降更新参数"""
        n_layers = len(self.hidden_layers) + 1
        for i in range(1, n_layers + 1):
            self.params[f"W{i}"] -= self.lr * grads[f"dW{i}"]
            self.params[f"b{i}"] -= self.lr * grads[f"db{i}"]

    def _onehot(self, y):
        y_idx = np.array([self.label_to_idx[label] for label in y])
        onehot = np.zeros((len(y), self.output_dim))
        onehot[np.arange(len(y)), y_idx] = 1
        return onehot, y_idx

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X = X_train.values if hasattr(X_train, "values") else X_train
        y = y_train.values if hasattr(y_train, "values") else y_train

        self.classes_ = sorted(np.unique(y).tolist())
        self.label_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        self.idx_to_label = {i: cls for cls, i in self.label_to_idx.items()}

        self.input_dim = X.shape[1]
        self.output_dim = len(self.classes_)
        self._init_weights(self.input_dim, self.output_dim)

        y_onehot, _ = self._onehot(y)
        n_samples = X.shape[0]
        n_layers = len(self.hidden_layers) + 1

        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.epochs):
            # Mini-batch 随机打乱
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch = X[batch_idx]
                y_batch = y_onehot[batch_idx]

                # 前向传播
                y_pred, cache = self._forward(X_batch, training=True)

                # 计算损失
                batch_loss = self._cross_entropy_loss(y_pred, y_batch)
                epoch_loss += batch_loss
                n_batches += 1

                # 反向传播
                grads = self._backward(cache, y_batch)

                # 更新参数
                self._update_params(grads)

            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_loss)

        self.is_trained = True
        return self

    def predict(self, X):
        X_arr = X.values if hasattr(X, "values") else X
        y_pred, _ = self._forward(X_arr, training=False)
        idx = np.argmax(y_pred, axis=1)
        return np.array([self.idx_to_label[i] for i in idx])

    def predict_proba(self, X):
        X_arr = X.values if hasattr(X, "values") else X
        y_pred, _ = self._forward(X_arr, training=False)
        return y_pred


def build(**kwargs):
    return HandWrittenANN(**kwargs)


AlgorithmFactory.register("ann", build)
