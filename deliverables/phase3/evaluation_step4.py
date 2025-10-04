#!/usr/bin/env python3
"""
Step 4 实验评估系统
基于现有代码：evaluation_clean.py、llm_pipeline和naive_rag_clean.py

实验设计：
- 实验组1：embedding size = 384 (sentence-transformers/all-MiniLM-L6-v2)
- 实验组2：embedding size = 512 (BAAI/bge-base-en-v1.5)
- 每组实验：k = 3, 5, 10
- 固定参数：seed=42, 样本数=5, 提示策略=basic
- 评估指标：f1-score和exact match
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple
import re
from naive_rag_clean import build_rag_database, load_rag_database
from llm_pipeline import LLM_Pipeline


class Step4Evaluator:
    """Step 4 实验评估器"""
    
    def __init__(self, seed: int = 42):
        """初始化评估器"""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # 实验配置
        self.experiment_configs = {
            "group1_384": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dim": 384,
                "k_values": [3, 5, 10]
            },
            "group2_512": {
                "model_name": "BAAI/bge-base-en-v1.5", 
                "embedding_dim": 512,
                "k_values": [3, 5, 10]
            }
        }
        
        # 固定参数
        self.num_samples = 5
        self.prompt_strategy = "basic"
        self.test_csv = "data/test.csv"
        self.training_csv = "data/training.csv"
    
    def calculate_f1_score(self, prediction: str, reference: str) -> float:
        """计算F1分数"""
        def normalize_text(text):
            text = text.lower().strip()
            text = re.sub(r'[^\w\s]', '', text)
            return text
        
        pred_tokens = normalize_text(prediction).split()
        ref_tokens = normalize_text(reference).split()
        
        if not ref_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        common_tokens = set(pred_tokens) & set(ref_tokens)
        
        if not pred_tokens:
            precision = 0.0
        else:
            precision = len(common_tokens) / len(pred_tokens)
        
        recall = len(common_tokens) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_exact_match(self, prediction: str, reference: str) -> float:
        """计算精确匹配"""
        pred_normalized = prediction.lower().strip()
        ref_normalized = reference.lower().strip()
        
        if pred_normalized == ref_normalized:
            return 1.0
        elif ref_normalized in pred_normalized:
            return 1.0
        else:
            return 0.0
    
    def get_llm_response(self, question: str, db: Any, k: int, strategy: str = "basic") -> str:
        """使用真实LLM获取响应"""
        try:
            # 初始化LLM Pipeline
            pipeline = LLM_Pipeline(
                rag_config_file="config_rag.yaml",
                llm_config_file="config_llm.yaml"
            )
            
            # 使用我们的SimpleVectorDB替换默认的向量数据库
            pipeline.vd = db
            
            # 生成上下文和答案
            context = pipeline.generate_context(question, k=k)
            answer = pipeline.generate_answer(
                query=question,
                strategy=strategy,
                k=k,
                temperature=0.1,
                max_tokens=150
            )
            
            return answer
            
        except Exception as e:
            print(f"    ⚠️ LLM Error: {e}")
            return f"Error generating response: {str(e)}"
    
    def run_single_experiment(self, model_name: str, embedding_dim: int, k: int) -> Dict[str, Any]:
        """运行单个实验"""
        print(f"\n🧪 Running Experiment: {model_name}, dim={embedding_dim}, k={k}")
        
        # 生成数据库目录名
        model_slug = model_name.replace("/", "_").replace("-", "_")
        db_dir = f"data/vector_db_emb{embedding_dim}_k{k}_{model_slug}"
        
        # 构建数据库
        print(f"🔨 Building database...")
        try:
            build_rag_database(
                model_name=model_name,
                training_csv=self.training_csv,
                output_dir=db_dir,
                tag=f"emb{embedding_dim}_k{k}"
            )
            print(f"✅ Database built successfully!")
        except Exception as e:
            print(f"❌ Error building database: {e}")
            return {"error": str(e)}
        
        # 加载数据库和测试数据
        print(f"🔧 Loading data...")
        try:
            db = load_rag_database(model_name, db_dir)
            test_data = pd.read_csv(self.test_csv)
            print(f"✅ Data loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return {"error": str(e)}
        
        # 采样测试数据
        sample_data = test_data.sample(n=self.num_samples, random_state=self.seed)
        print(f"📊 Testing with {len(sample_data)} questions")
        
        # 运行测试
        detailed_results = []
        f1_scores = []
        exact_matches = []
        
        for idx, (_, row) in enumerate(sample_data.iterrows()):
            question = row['question']
            true_answer = row['answer']
            
            print(f"  Q{idx+1}: {question[:50]}...")
            
            # 使用真实LLM获取响应
            print(f"    🤖 Calling LLM...")
            predicted_answer = self.get_llm_response(question, db, k, self.prompt_strategy)
            
            # 获取检索到的文档用于记录
            results = db.search(question, k=k)
            retrieved_passages = [r['passage'] for r in results['results']]
            context = " ".join(retrieved_passages)
            
            # 创建提示词记录（用于保存到CSV）
            prompt = f"""Based on the following context, please answer the question.
Please provide only a short, direct answer without any explanation or additional information.

Context: {context}

Question: {question}

Answer:"""
            
            # 计算指标
            f1_score = self.calculate_f1_score(predicted_answer, true_answer)
            exact_match = self.calculate_exact_match(predicted_answer, true_answer)
            
            f1_scores.append(f1_score)
            exact_matches.append(exact_match)
            
            # 保存详细结果
            detailed_results.append({
                "question": question,
                "true_answer": true_answer,
                "predicted_answer": predicted_answer,
                "prompt": prompt,
                "f1_score": f1_score,
                "exact_match": exact_match,
                "k": k,
                "embedding_dim": embedding_dim,
                "model_name": model_name
            })
            
            print(f"    F1: {f1_score:.3f}, EM: {exact_match:.3f}")
        
        # 计算平均指标
        avg_f1 = np.mean(f1_scores)
        avg_exact_match = np.mean(exact_matches)
        
        print(f"📊 Results: F1={avg_f1:.4f}, EM={avg_exact_match:.4f}")
        
        return {
            "detailed_results": detailed_results,
            "avg_f1": avg_f1,
            "avg_exact_match": avg_exact_match,
            "f1_scores": f1_scores,
            "exact_matches": exact_matches,
            "config": {
                "model_name": model_name,
                "embedding_dim": embedding_dim,
                "k": k,
                "num_samples": self.num_samples,
                "seed": self.seed
            }
        }
    
    def save_detailed_results(self, results: Dict[str, Any], group_name: str, k: int) -> str:
        """保存详细结果CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detailed_results_{group_name}_k{k}_{timestamp}.csv"
        
        df = pd.DataFrame(results["detailed_results"])
        df.to_csv(filename, index=False)
        print(f"✅ Detailed results saved to: {filename}")
        return filename
    
    def save_summary_results(self, group_results: List[Dict[str, Any]], group_name: str) -> str:
        """保存摘要结果CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_results_{group_name}_{timestamp}.csv"
        
        summary_data = []
        for result in group_results:
            summary_data.append({
                "k": result["config"]["k"],
                "avg_f1_score": result["avg_f1"],
                "avg_exact_match": result["avg_exact_match"],
                "model_name": result["config"]["model_name"],
                "embedding_dim": result["config"]["embedding_dim"],
                "num_samples": result["config"]["num_samples"],
                "seed": result["config"]["seed"]
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(filename, index=False)
        print(f"✅ Summary results saved to: {filename}")
        return filename
    
    def run_experiment_group(self, group_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """运行实验组"""
        print(f"\n🚀 Running Experiment Group: {group_name}")
        print(f"   Model: {config['model_name']}")
        print(f"   Embedding dim: {config['embedding_dim']}")
        print(f"   K values: {config['k_values']}")
        
        group_results = []
        
        for k in config['k_values']:
            result = self.run_single_experiment(
                model_name=config['model_name'],
                embedding_dim=config['embedding_dim'],
                k=k
            )
            
            if "error" not in result:
                # 保存详细结果
                detailed_file = self.save_detailed_results(result, group_name, k)
                result["detailed_file"] = detailed_file
                group_results.append(result)
            else:
                print(f"❌ Experiment failed: {result['error']}")
        
        # 保存摘要结果
        if group_results:
            summary_file = self.save_summary_results(group_results, group_name)
            print(f"📊 Group {group_name} completed!")
            print(f"   Summary: {summary_file}")
            
            # 显示结果对比
            print(f"📈 Results Comparison for {group_name}:")
            for result in group_results:
                print(f"   K={result['config']['k']}: F1={result['avg_f1']:.4f}, EM={result['avg_exact_match']:.4f}")
        
        return group_results
    
    def run_all_experiments(self) -> Dict[str, List[Dict[str, Any]]]:
        """运行所有实验"""
        print("🎯 Starting Step 4 Experiments")
        print("=" * 60)
        print(f"📋 Configuration:")
        print(f"   Seed: {self.seed}")
        print(f"   Samples per experiment: {self.num_samples}")
        print(f"   Prompt strategy: {self.prompt_strategy}")
        print(f"   Test data: {self.test_csv}")
        print()
        
        all_results = {}
        
        for group_name, config in self.experiment_configs.items():
            group_results = self.run_experiment_group(group_name, config)
            all_results[group_name] = group_results
        
        # 生成最终汇总报告
        self.generate_final_report(all_results)
        
        return all_results
    
    def generate_final_report(self, all_results: Dict[str, List[Dict[str, Any]]]):
        """生成最终汇总报告"""
        print(f"\n📊 Final Experiment Summary")
        print("=" * 50)
        
        for group_name, group_results in all_results.items():
            print(f"\n{group_name.upper()}:")
            for result in group_results:
                print(f"  K={result['config']['k']}: F1={result['avg_f1']:.4f}, EM={result['avg_exact_match']:.4f}")
        
        # 保存最终汇总
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_summary_file = f"step4_final_summary_{timestamp}.csv"
        
        final_data = []
        for group_name, group_results in all_results.items():
            for result in group_results:
                final_data.append({
                    "group": group_name,
                    "model_name": result["config"]["model_name"],
                    "embedding_dim": result["config"]["embedding_dim"],
                    "k": result["config"]["k"],
                    "avg_f1_score": result["avg_f1"],
                    "avg_exact_match": result["avg_exact_match"],
                    "num_samples": result["config"]["num_samples"],
                    "seed": result["config"]["seed"]
                })
        
        df = pd.DataFrame(final_data)
        df.to_csv(final_summary_file, index=False)
        print(f"\n✅ Final summary saved to: {final_summary_file}")


def main():
    """主函数"""
    print("🚀 Step 4 Experiment Evaluation System")
    print("=" * 60)
    
    # 创建评估器
    evaluator = Step4Evaluator(seed=42)
    
    # 运行所有实验
    results = evaluator.run_all_experiments()
    
    print(f"\n✅ All experiments completed successfully!")
    print(f"📁 Check generated CSV files for detailed results")


if __name__ == "__main__":
    main()
