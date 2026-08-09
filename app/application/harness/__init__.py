# -*- coding: utf-8 -*-
"""harness

Agent 运行时护栏的判定层（纯逻辑，不碰 IO）：

    loop_detector   14 章  滑动窗口循环检测——同一工具反复横跳时注入收敛提示
    assertions      17-3 章 三类单步断言（Schema / Sequencing / Semantic）
    drift_detector  17-3 章 Silent-Drift 静默漂移检测

这些判定由 app/infrastructure/harness_middleware.py 在工具边界调用，
由 app/application/agents/orchestrator.py 在轮次边界调用。
判定层不依赖 AgentScope，可单独单测。
"""
