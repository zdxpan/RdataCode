class Node(object):
    def __init__(self, data = -1, lchild = None, rchild = None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild

class BinaryTree(object):
    def __init__(self):
        self.root = Node()

    def add(self, data):
        node = Node(data)
        if self.isEmpty():
            self.root = node
        else:
            tree_node = self.root
            queue = []
            queue.append(self.root)

            while queue:
                tree_node = queue.pop(0)
                if tree_node.lchild == None:
                    tree_node.lchild = node
                    return
                elif tree_node.rchild == None:
                    tree_node.rchild = node
                    return
                else:
                    queue.append(tree_node.lchild)
                    queue.append(tree_node.rchild)

    def pre_order(self, start):
        node = start
        if node == None:
            return

        print( node.data)
        if node.lchild == None and node.rchild == None:
            return
        self.pre_order(node.lchild)
        self.pre_order(node.rchild)

    def pre_order_loop(self):
        if self.isEmpty():
            return

        stack = []
        node = self.root
        while node or stack:
            while node:
                print( node.data)
                stack.append(node)
                node = node.lchild
            if stack:
                node = stack.pop()
                node = node.rchild

    def in_order(self, start):
        node = start
        if node == None:
            return
        self.in_order(node.lchild)
        print( node.data)
        self.in_order(node.rchild)

    def in_order_loop(self):
        if self.isEmpty():
            return
        
        stack = []
        node = self.root
        while node or stack:
            while node:
                stack.append(node)
                node = node.lchild

            if stack:
                node = stack.pop()
                print( node.data)
                node = node.rchild

    def post_order(self, start):
        node = start
        if node == None:
            return
        self.post_order(node.lchild)
        self.post_order(node.rchild)
        print( node.data)

    
    def post_order_loop(self):
        if self.isEmpty():
            return
        
        node = self.root
        stack = []
        queue = []
        queue.append(node)
        while queue:
            node = queue.pop()
            if node.lchild:
                queue.append(node.lchild)
            if node.rchild:
                queue.append(node.rchild)
            stack.append(node)
        while stack:
            print( stack.pop().data)

    #if lchild and rchild are None or lchild and rchild are printed, print the parent node node and pop out of the stack
    #else lchild and rchild push into the stack
    def post_order_loop1(self):
        if self.isEmpty():
            return

        stack = []
        top = -1
        node = self.root
        stack.append(node)
        #we need to recognize the last printed node
        top += 1
        pre = None
        while stack:
            node = stack[-1]
            if node.lchild is None and node.rchild is None:
                print( node.data)
                pre = node
                top -= 1
            elif not pre and (node.lchild == pre or node.rchild == pre):
                print( node.data)
                pre = node
                top -= 1
            else:
                if node.rchild:
                    if top < len(stack)-1:
                        stack[top] = node.rchild
                    else:
                        stack.append(node.rchild)
                if node.lchild:
                    if top < len(stack)-1:
                        stack[top] = node.lchild
                    else:
                        stack.append(node.lchild)

    def level_order(self):
        node = self.root
        if node == None:
            return
        
        queue = []
        queue.append(node)

        while queue:
            node = queue.pop(0)
            print( node.data)
            if node.rchild:
                queue.append(node.rchild)
            if node.lchild:
                queue.append(node.lchild)
        print()

    def isEmpty(self):
        return True if self.root.data == -1 else False

if __name__ == '__main__':
    arr = []
    for i in range(10):
        arr.append(i)
    print( arr)

    tree = BinaryTree()
    for i in arr:
        tree.add(i)
    print( 'level_order:')
    tree.level_order()
    print( 'pre order:')
    tree.pre_order(tree.root)
    print( '\npre order loop:')
    tree.pre_order_loop()
    print( '\nin_order:')
    tree.in_order(tree.root)
    print('\nin_order loop:')
    tree.in_order_loop()
    print( '\npost_order:')
    tree.post_order(tree.root)
    print( '\npost_order_loop:')
    tree.post_order_loop()
    print( '\npost_order_loop1:')
    tree.post_order_loop1()
    
    
    
    
    
    
    
    
    
    
    
    
    
#-------------------
'invert a binary tree'    
class solution:
    def invertTree(self,root):
        if root:
            invert = self.invertTree
            root.left,root.right = invert(root.right),invert(root.left)+
            #root.left,root.right = self.invertTree(root.right),self.invertTree(root.left)
            return root
        
def invertTree(self, root):
    stack = [root]
    while stack:
        node = stack.pop()#默认弹出倒数第一个元素
        if node:
            node.left, node.right = node.right, node.left
            stack += node.left, node.right
    return root
    
#Java solutiom
public class solution{
        public TreeNode invertTree(TreeNode root){
                if(root == null){
                        return null                      
                }
                final TreeNode left = root.left,right = root.right;
                root.left = invertTree(right);
                root.right = invertTree(left)
                return root#这是迭代最后时候得到的结果，
            }
        }    
    
'Validate Binary Search Tree'
1 inorder --> check array sorted  o(N) time
2 getMin / getMax(root.left) < root.val and getMin(root.right) > root.val --> recursion
class solution(object):
    def isValidBST(self,root):
        
        if root.val >= root.left.val: return False
        if root.val <= root.right.val: return False
        return isValidBST(root.left) and isValidBST(root.right)
    
    
    
    
class solution:
    long pre = long.MIN_VALUE
    public boolean isValidBST(TreeNode root)
        if root:
            if not isValidBST(root.left):return False
            if root.val <= pre:return False
            pre = root.val
            if not isValidBST(root.right) :return False
            return True
            
'------------------------中序遍历，判断是否有序，或者判断是否跟sorted的一样------------------------------------------'
class Solution:
    # @param root, a tree node
    # @return a boolean
    # 7:38
    def isValidBST(self, root):
        output = []
        self.inOrder(root, output)
        
        for i in range(1, len(output)):
            if output[i-1] >= output[i]:
                return False

        return True
    #--------------way ||
        self.res = list()
        self.validation(root)
        return self.res == sorted(self.res) and len(set(self.res)) == len(self.res)

    def inOrder(self, root, output):
        if root is None:
            return
        
        self.inOrder(root.left, output)
        output.append(root.val)
        self.inOrder(root.right, output)
'-------------------------------------------------------------------------'
'1  数深度 max depth of binary tree'
class Solution(object):
    def maxDepth(self,root):
        
        if not root:return 0
        if not root.left and not root.right : return 1#理解就好
        #lefDepth = self.maxDepth(root.left)
        #rightDepth = self.maxDepth(root.right)
        #return max(leftDepth,rightDepth)+1
        return max(self.maxDepth(root.left),self.maxDepth(root.right))+1
    
    
'convert sorted list to binary search tree'
'given an singly linked list where element are sorted in ascending order ,convert'
'it into a height banlanced bst'
public class Solution{

public TreeNode sortedListToBST(listNode head){
        if head == NULL :return NULL
        return toBST(head,null)
}
public TreeNode toBST(ListNode head,ListNode tail){
        ListNode slow = head;
        ListNode fast = head;
        if head ==tail :return NULL
        
        while fast!=tail and fast.next! = tail:
            fast = fast.next.next
            slow = slow.next
        
        TreeNode thead = new TreeNode(slow.val)
        thead.left = toBST(head,slow)
        thead.right = toBST(slow.next,tail)
        return thead
        
}
        
}    
    

链接：https://www.nowcoder.com/questionTerminal/86343165c18a4069ab0ab30c32b1afd0
来源：牛客网
'''链接：https://www.nowcoder.com/questionTerminal/86343165c18a4069ab0ab30c32b1afd0
来源：牛客网

由于二分查找法每次需要找到中点，而链表的查找中间点可以通过快慢指针来操作。找到中点后，
要以中点的值建立一个数的根节点，然后需要把原链表断开，分为前后两个链表，都不能包含原中节点，
然后再分别对这两个链表递归调用原函数，分别连上左右子节点即可'''
public TreeNode sortedListToBST(ListNode head) {
        return toBST(head, null);
    }
 
    private TreeNode toBST(ListNode head, ListNode tail) {
        if (head == tail)
            return null;
        // 申请两个指针，fast移动速度是low的两倍
        ListNode fast = head;
        ListNode slow = head;
        while (fast != tail && fast.next != tail) {
            fast = fast.next.next;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);#找到中值
        root.left = toBST(head, slow);
        root.right = toBST(slow.next, tail);
 
        return root;
}
public class Solution {
    ListNode cur;
    public TreeNode sortedListToBST(ListNode head) {  
        cur = head;
        int len = 0;
        while (head != null) {
            head = head.next;
            len++;
        }
        return helper(0, len-1);
    }
    public TreeNode helper(int start, int end) {
        if (start > end) return null;
        int mid = start+(end-start)/2;
        TreeNode left = helper(start, mid-1);
        TreeNode root = new TreeNode(cur.val);
        cur = cur.next;
        TreeNode right = helper(mid+1, end);
        root.left = left;
        root.right = right;
        return root;
    }
}        
'Lowest Common Ancestor of a Binary Search Tree '
如果是二叉搜索树，看什么时候劈腿，则该节点就是共有祖先    
    
def lowestCommonAncestor(self, root, p, q):
    while (root.val - p.val ) * (root.val - q.val) > 0:
        if root.val - p > 0 :
            root = root.left
        else:
            root = root.right
        #root = (root.left,root.right)[root.val < p.val]
    return root
        
    
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


'-------------镜像对称二叉树---------------------'
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return not root or isMirror(root.left,root.right)
    
    def isMirror(p,q):
        if not p and not q:return True
        if not p or not q:return False
        return p.val == q.val and isMirror(p.left,q.right) and isMirror(p.right,q.left)



class Solution:
    def flatten(root):
        TreNode prev
        if root:
            flatten(root.right)
            flatten(root.left)
            root.right = prev
            root.left = null
            pre = root

























    