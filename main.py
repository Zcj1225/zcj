import operator
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation


# ==========================================
# 1. 定义系统状态 (State)：作为各 Agent 之间传递的“共享内存”
# ==========================================
class AgentState(TypedDict):
    topic: str  # 用户输入的宏观研究主题
    sub_tasks: str  # Manager 拆解出的子任务
    research_data: str  # Researcher 收集到的底层数据
    draft: str  # Writer 生成的研报初稿
    review_feedback: str  # Reviewer 给出的审查意见
    iteration_count: int  # 记录打回重做的次数，防止无限死循环


# 初始化大模型 (这里使用 GPT-4o 作为底层推理引擎)
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


# ==========================================
# 2. 定义各个 Agent 节点 (Nodes)
# ==========================================

def manager_agent(state: AgentState):
    """Manager: 负责意图解析与任务拆解"""
    print("--- 🤖 [Manager Agent] 正在拆解宏观任务 ---")
    prompt = f"你是一个资深研究主管。请将以下研究主题拆解为3-4个具体的数据检索子任务。\n主题: {state['topic']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"sub_tasks": response.content, "iteration_count": 0}


def researcher_agent(state: AgentState):
    """Researcher: 负责动态检索与数据挖掘 (这里省略了真实的 Search API 调用，用 Prompt 模拟)"""
    print("--- 🔍 [Researcher Agent] 正在执行数据挖掘 ---")
    # 在真实项目中，这里会结合 SerpAPI, 数据库 SQL 工具等进行 ReAct 循环检索
    prompt = f"根据以下子任务，总结并提供相关的市场数据、行业报告和事实依据。\n任务: {state['sub_tasks']}"

    # 模拟增量检索：如果包含之前的审查意见，则针对性补充数据
    if state.get("review_feedback"):
        prompt += f"\n请特别注意并补充解决以下审查意见中提到的数据缺失或逻辑漏洞：{state['review_feedback']}"

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"research_data": response.content}


def writer_agent(state: AgentState):
    """Writer: 负责结构化生成研报初稿"""
    print("--- ✍️ [Writer Agent] 正在撰写研报初稿 ---")
    prompt = f"""你是一个专业金融分析师。请根据以下研究数据，撰写一份严谨的行业分析初稿。
    研究主题: {state['topic']}
    研究数据: {state['research_data']}
    要求：结构清晰，包含摘要、正文和风险提示。"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"draft": response.content}


def reviewer_agent(state: AgentState):
    """Reviewer (Critic): 负责交叉验证与逻辑思辨 (红方对抗)"""
    print("--- 🧐 [Reviewer Agent] 正在进行逻辑审查 ---")
    prompt = f"""你是一个苛刻的首席审查官。请审查以下研报初稿，寻找逻辑断层、数据幻觉或论证不足的地方。
    研究主题: {state['topic']}
    初稿内容: {state['draft']}
    如果你认为文章已经达标，请回复“APPROVED”。否则，请给出具体的修改意见。"""

    response = llm.invoke([HumanMessage(content=prompt)])
    feedback = response.content

    # 提取当前迭代次数
    current_count = state.get("iteration_count", 0)

    return {"review_feedback": feedback, "iteration_count": current_count + 1}


# ==========================================
# 3. 定义路由逻辑 (Conditional Edges)
# ==========================================

def route_after_review(state: AgentState):
    """决定是结束工作流，还是打回重做"""
    feedback = state.get("review_feedback", "")
    iteration_count = state.get("iteration_count", 0)

    # 如果审核通过，或者已经重做超过2次（防止死循环），则结束
    if "APPROVED" in feedback or iteration_count >= 3:
        print("✅ 审核通过 或 达到最大迭代次数，流程结束。")
        return "end"
    else:
        print(f"⚠️ 发现逻辑漏洞，打回重做！(当前重做次数: {iteration_count})")
        # 将工作流导回 Researcher，让其根据 feedback 重新找数据
        return "researcher"

    # ==========================================


# 4. 构建并编译多 Agent 工作流图 (Graph)
# ==========================================

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("manager", manager_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("reviewer", reviewer_agent)

# 定义固定边 (数据流转顺序)
workflow.set_entry_point("manager")
workflow.add_edge("manager", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")

# 添加条件边 (引入长链推理的打回重做机制)
workflow.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "end": END,
        "researcher": "researcher"  # 驳回时回到检索节点
    }
)

# 编译图
app = workflow.compile()

# ==========================================
# 5. 触发执行
# ==========================================
if __name__ == "__main__":
    initial_topic = "2026年全球固态电池行业现状及头部企业(宁德时代vs丰田)对比分析"
    print(f"🚀 开始执行研究任务: {initial_topic}\n")

    # 传入初始状态启动工作流
    final_state = app.invoke({"topic": initial_topic})

    print("\n" + "=" * 50)
    print("🎉 最终研报产出：\n")
    print(final_state.get("draft"))
    print("=" * 50)
