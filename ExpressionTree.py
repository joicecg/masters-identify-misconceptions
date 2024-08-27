class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class ExpressionTree:
    def parse_equation_to_tree(equation):
        # Split the equation into left and right parts
        left_part, right_part = equation.split('=')
        
        def construct_tree(expression):
            elements = expression.split()
            root = None
            current_node = None
            
            for element in elements:
                if element in "()": # Skip parentheses
                    continue
                if element.isdigit() or element.isalpha():  # Operand
                    new_node = TreeNode(element)
                    if not root:
                        root = new_node
                    else:
                        current_node.right = new_node
                else:  
                    new_node = TreeNode(element)
                    new_node.left = root
                    root = new_node
                current_node = new_node
            
            return root
        
        # Construct trees for both parts
        left_tree = construct_tree(left_part)
        right_tree = construct_tree(right_part)
        
        return left_tree, right_tree

    # Simple function to print the tree in pre-order traversal (root, left, right)
    def print_tree(node, level=0):
        if node is not None:
            print(' ' * level + str(node.value))
            ExpressionTree.print_tree(node.left, level + 2)
            ExpressionTree.print_tree(node.right, level + 2)

# Function for inorder traversal of the tree to generate a string representation
def inorder_to_string(node):
    if node is None:
        return ""
    if node.left is None and node.right is None:
        return str(node.value)
    left_str = inorder_to_string(node.left)
    right_str = inorder_to_string(node.right)
    if left_str:  
        left_str = f"{left_str}"
    if right_str:
        right_str = f"{right_str}"
    return f"{left_str} {node.value} {right_str}"
