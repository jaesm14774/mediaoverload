import json
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich import print_json
from typing import Dict, Any

info = {
    'basic': '打印基本摘要，包括節點總數等簡單資訊。',
    'all_node_detail': '打印所有節點的詳細信息。',
    'specific_node_detail': '打印特定類型節點的詳細信息，例如 CLIPTextEncode 節點。',
    'node_connection': '打印節點之間的連接關係。',
    'json_print': '使用 rich.print_json 打印美化的 JSON。',
}

console = Console()

# 創建表格
table = Table(title="analyze_workflow 可用的 print_type 選項", title_style="bold cyan")
table.add_column("Type", style="bold green", justify="left")
table.add_column("Description", style="bright_black", justify="left")

# 填入數據
for key, desc in info.items():
    table.add_row(key, desc)

# console.print(table)

class WorkflowAnalyzer:
    @staticmethod
    def print_workflow_summary(workflow: Dict[str, Any]) -> None:
        """使用rich庫打印工作流摘要"""
        console = Console()
        
        # 創建主表格
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Node Type", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Node IDs", style="blue")
        
        # 收集節點類型統計
        node_types = {}
        for node_id, node_data in workflow.items():
            node_type = node_data.get("class_type", "Unknown")
            if node_type not in node_types:
                node_types[node_type] = {"count": 0, "ids": []}
            node_types[node_type]["count"] += 1
            node_types[node_type]["ids"].append(node_id)
        
        # 填充表格
        for node_type, info in sorted(node_types.items()):
            table.add_row(
                node_type,
                str(info["count"]),
                ", ".join(info["ids"])
            )
        
        console.print("\n[bold yellow]Workflow Summary:[/bold yellow]")
        console.print(table)

    @staticmethod
    def print_node_details(workflow: Dict[str, Any], node_type: str = None) -> None:
        """
        打印節點的詳細資訊，包括：
        - 類型和ID
        - 輸入參數
        - 輸出連接
        - 元數據
        - 其他相關資訊
        """
        console = Console()
        tree = Tree("[bold yellow]Workflow Nodes[/bold yellow]")
        
        def format_value(value: Any) -> str:
            """格式化顯示值"""
            if isinstance(value, list):
                if len(value) == 2 and all(isinstance(x, (int, str)) for x in value):
                    # 這是一個節點連接
                    source_id, output_index = value
                    source_type = workflow.get(str(source_id), {}).get("class_type", "Unknown")
                    return f"[blue]Connected to[/blue] {source_type} ({source_id})[dim]:{output_index}[/dim]"
                return str(value)
            return str(value)
    
        def add_dict_to_tree(branch, data: Dict, style: str = "yellow"):
            """遞迴添加字典內容到樹狀結構"""
            for key, value in data.items():
                if isinstance(value, dict):
                    sub_branch = branch.add(f"[{style}]{key}[/{style}]")
                    add_dict_to_tree(sub_branch, value, style)
                else:
                    branch.add(f"[{style}]{key}[/{style}]: {format_value(value)}")
    
        for node_id, node_data in workflow.items():
            current_type = node_data.get("class_type", "Unknown")
            
            # 如果指定了節點類型，只顯示該類型
            if node_type and current_type != node_type:
                continue
            
            # 創建主節點分支
            node_branch = tree.add(
                f"[bold cyan]{current_type}[/bold cyan] ([blue]{node_id}[/blue])"
            )
            
            # 1. 添加輸入參數
            if "inputs" in node_data:
                inputs_branch = node_branch.add("[green]Inputs[/green]")
                add_dict_to_tree(inputs_branch, node_data["inputs"])
            
            # 2. 添加特定類型節點的詳細資訊
            if current_type == "CLIPTextEncode":
                # CLIP文本編碼器特有資訊
                clip_branch = node_branch.add("[magenta]CLIP Info[/magenta]")
                if "clip" in node_data.get("inputs", {}):
                    clip_source = node_data["inputs"]["clip"]
                    if isinstance(clip_source, list):
                        source_id, output_index = clip_source
                        clip_model = workflow.get(str(source_id), {}).get("class_type", "Unknown")
                        clip_branch.add(f"Model: {clip_model} ({source_id})[dim]:{output_index}[/dim]")
            
            elif current_type == "KSampler":
                # KSampler特有資訊
                sampler_branch = node_branch.add("[magenta]Sampler Info[/magenta]")
                inputs = node_data.get("inputs", {})
                sampler_branch.add(f"Method: {inputs.get('sampler_name', 'Unknown')}")
                sampler_branch.add(f"Scheduler: {inputs.get('scheduler', 'Unknown')}")
                sampler_branch.add(f"Steps: {inputs.get('steps', 'Unknown')}")
                sampler_branch.add(f"CFG: {inputs.get('cfg', 'Unknown')}")
                sampler_branch.add(f"Denoise: {inputs.get('denoise', 'Unknown')}")
            
            elif current_type == "CheckpointLoaderSimple":
                # 檢查點加載器特有資訊
                checkpoint_branch = node_branch.add("[magenta]Checkpoint Info[/magenta]")
                checkpoint_branch.add(f"Model: {node_data.get('inputs', {}).get('ckpt_name', 'Unknown')}")
            
            elif current_type == "VAEDecode":
                # VAE解碼器特有資訊
                vae_branch = node_branch.add("[magenta]VAE Info[/magenta]")
                if "vae" in node_data.get("inputs", {}):
                    vae_source = node_data["inputs"]["vae"]
                    if isinstance(vae_source, list):
                        source_id, output_index = vae_source
                        vae_model = workflow.get(str(source_id), {}).get("class_type", "Unknown")
                        vae_branch.add(f"Model: {vae_model} ({source_id})[dim]:{output_index}[/dim]")
            
            elif current_type == "EmptyLatentImage":
                # 空白潛空間圖像特有資訊
                latent_branch = node_branch.add("[magenta]Latent Info[/magenta]")
                inputs = node_data.get("inputs", {})
                latent_branch.add(f"Width: {inputs.get('width', 'Unknown')}")
                latent_branch.add(f"Height: {inputs.get('height', 'Unknown')}")
                latent_branch.add(f"Batch Size: {inputs.get('batch_size', 'Unknown')}")
            
            # 3. 添加輸出連接資訊
            output_connections = []
            for other_id, other_data in workflow.items():
                for input_name, input_value in other_data.get("inputs", {}).items():
                    if isinstance(input_value, list) and str(input_value[0]) == node_id:
                        output_connections.append((other_id, other_data["class_type"], input_name, input_value[1]))
            
            if output_connections:
                outputs_branch = node_branch.add("[red]Connected To[/red]")
                for target_id, target_type, input_name, output_index in output_connections:
                    outputs_branch.add(
                        f"[yellow]{target_type}[/yellow] ({target_id}) "
                        f"[dim]via {input_name} (output {output_index})[/dim]"
                    )
            
            # 4. 添加其他元數據
            if "_meta" in node_data:
                metadata_branch = node_branch.add("[cyan]Metadata[/cyan]")
                add_dict_to_tree(metadata_branch, node_data["_meta"], "cyan")
        
        console.print("\n")
        console.print(tree)

    @staticmethod
    def print_node_connections(workflow: Dict[str, Any]) -> None:
        """打印節點之間的連接關係"""
        console = Console()
        tree = Tree("[bold yellow]Node Connections[/bold yellow]")
        
        for node_id, node_data in workflow.items():
            node_type = node_data.get("class_type", "Unknown")
            node_branch = tree.add(f"[bold cyan]{node_type}[/bold cyan] ([blue]{node_id}[/blue])")
            
            if "inputs" in node_data:
                inputs_branch = node_branch.add("[green]Connected From[/green]")
                for input_name, input_value in node_data["inputs"].items():
                    if isinstance(input_value, list) and len(input_value) == 2:
                        source_id, output_index = input_value
                        source_type = workflow.get(str(source_id), {}).get("class_type", "Unknown")
                        inputs_branch.add(
                            f"[yellow]{input_name}[/yellow] <- {source_type} ({source_id})[dim]:{output_index}[/dim]"
                        )
        
        console.print("\n")
        console.print(tree)

# 使用範例
def analyze_workflow(workflow_path=None, workflow=None, node_type="CLIPTextEncode", print_type='basic'):
    """分析並顯示工作流信息"""
    print_type = print_type.lower()

    if not workflow:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
    analyzer = WorkflowAnalyzer()
    
    # 1. 打印基本摘要
    if print_type == 'basic':
        analyzer.print_workflow_summary(workflow)
    
    # 2. 打印所有節點詳細信息
    if print_type == 'all_node_detail':
        analyzer.print_node_details(workflow)
    
    # 3. 打印特定類型節點的詳細信息
    if print_type == 'specific_node_detail':
        print("\n[bold yellow]CLIPTextEncode Nodes Details:[/bold yellow]")
        analyzer.print_node_details(workflow, node_type)
    
    # 4. 打印節點連接關係
    if print_type == 'node_connection':
        analyzer.print_node_connections(workflow)
    
    # 5. 使用 rich.print_json 直接打印美化的 JSON
    if print_type == 'json_print':
        print("\n[bold yellow]Raw JSON (Formatted):[/bold yellow]")
        print_json(data=workflow)
