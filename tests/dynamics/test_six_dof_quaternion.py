"""6-DOF 四元数工具函数单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.dynamics.six_dof import (
    _axis_from_quat,
    _normalize_quat,
    _quat_from_vectors,
    _quat_multiply,
    _quat_rotate,
)


def test_quat_multiply_identity() -> None:
    """四元数乘法单位元：q * [0,0,0,1] == q。"""
    q = np.array([0.1, 0.2, 0.3, 0.4])
    q = q / np.linalg.norm(q)
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(_quat_multiply(q, identity), q, atol=1e-12)
    assert np.allclose(_quat_multiply(identity, q), q, atol=1e-12)


def test_quat_rotate_90_degrees_around_z() -> None:
    """绕 z 轴旋转 90° 将 x 轴映射到 y 轴。"""
    angle = np.deg2rad(90.0)
    q = np.array([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)])
    v = np.array([1.0, 0.0, 0.0])
    rotated = _quat_rotate(q, v)
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


def test_quat_rotate_preserves_norm() -> None:
    """四元数旋转保持向量模长。"""
    q = np.array([0.2, -0.3, 0.1, 0.9])
    q = q / np.linalg.norm(q)
    v = np.array([3.0, -4.0, 5.0])
    rotated = _quat_rotate(q, v)
    assert np.linalg.norm(rotated) == pytest.approx(np.linalg.norm(v), rel=1e-12)


def test_axis_from_quat_identity() -> None:
    """单位四元数对应的弹轴为 x 轴。"""
    q = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(_axis_from_quat(q), [1.0, 0.0, 0.0], atol=1e-12)


def test_quat_from_vectors_basic() -> None:
    """从 x 轴构造到 y 轴的四元数，旋转后应得到 y 轴。"""
    q = _quat_from_vectors(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    rotated = _quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


def test_quat_from_vectors_parallel() -> None:
    """两向量平行时返回单位四元数。"""
    q = _quat_from_vectors(np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]))
    assert np.allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_normalize_quat_zero() -> None:
    """零四元数归一化后返回单位四元数。"""
    assert np.allclose(_normalize_quat(np.zeros(4)), [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_normalize_quat_non_zero() -> None:
    """非零四元数归一化后模长为 1。"""
    q = np.array([1.0, 2.0, 3.0, 4.0])
    qn = _normalize_quat(q)
    assert np.linalg.norm(qn) == pytest.approx(1.0, rel=1e-12)
    assert np.allclose(qn, q / np.linalg.norm(q), atol=1e-12)
