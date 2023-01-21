//
// Created by Nathan on 2023-01-17.
//

#ifndef DEV_QHC_CPP_PROJECTS_TERNARYTRIES_HPP
#define DEV_QHC_CPP_PROJECTS_TERNARYTRIES_HPP

#include <string>
#include "lib_central/utils.h"

using std::string;

/**
 *
 * @tparam T_Val copyable
 */
template<typename T_Val>
class TernaryTries final {
public:
    class Node;

private:

    enum class Direction {
        LEFT,
        MID,
        RIGHT,
        NONE
    };

    Node *root{nullptr};
    int count{};

    bool get_value(Node *n, const string &key, int depth, T_Val *ret) {
        if (n == nullptr) {
            return false;
        }
        char c = key.at(depth);
        if (c < n->chr) { return get_value(n->left, key, depth); }
        else if (c > n->chr) { return get_value(n->right, key, depth); }
        else if (depth < key.length() - 1) { return get_value(n->mid, key, depth + 1); }
        else {
            if (ret != nullptr) {
                *ret = n->val;
            }
            return true;
        }
    }

    static void recursive_release(Node *n) noexcept {
        if (n->left != nullptr) {
            recursive_release(n->left);
        }
        if (n->mid != nullptr) {
            recursive_release(n->mid);
        }
        if (n->right != nullptr) {
            recursive_release(n->right);
        }
        delete n;
    }

    Node *insert_node(Node *n, const string &key, const T_Val &val, int depth, bool replace) {
        char c = key.at(depth);
        if (n == nullptr) {
            n = new Node{};
            n->chr = c;
        }
        if (c < n->c) { n->left = insert_node(n->left, key, val, depth, replace); }
        else if (c > n->c) { n->right = insert_node(n->right, key, val, depth, replace); }
        else if (depth < key.length() - 1) { n->mid = insert_node(n->mid, key, val, depth + 1); }
        else {
            if (!replace && n->contain) {
                return n;
            } else {
                n->val = val;
                n->contain = true;
                count++;
            }
        }
        return n;
    }

    void remove_dangle_node(Node *n, Node *p, Direction direct) {
        if (!n->contain && n->left == nullptr && n->mid == nullptr && n->right == nullptr && p != nullptr) {
            switch (direct) {
                case Direction::MID:
                    delete p->mid;
                    p->mid = nullptr;
                    break;
                case Direction::LEFT:
                    delete p->left;
                    p->left = nullptr;
                    break;
                case Direction::RIGHT:
                    delete p->right;
                    p->right = nullptr;
                    break;
            }
        }
        if (p == nullptr && !n->contain && n->left == nullptr && n->mid == nullptr && n->right == nullptr) {
            delete root;
            root = nullptr;
        }
    }

    bool remove_node(Node *n, Node *p, Direction direct, const string &key, int d, T_Val *ret) {
        if (n == nullptr) { return false; }
        char c = key.at(d);
        if (c < n->chr) {
            bool b{remove_node(n->left, n, Direction::LEFT, key, d, ret)};
            remove_dangle_node(n, p, direct);
            return b;
        } else if (c > n->chr) {
            bool b{remove_node(n->right, n, Direction::RIGHT, key, d, ret)};
            remove_dangle_node(n, p, direct);
            return b;
        } else if (d < key.length() - 1) {
            bool b{remove_node(n->mid, n, Direction::MID, key, d + 1, ret)};
            remove_dangle_node(n, p, direct);
            return b;
        } else if (n->contain) {
            n->contain = false;
            remove_dangle_node(n, p, direct);
            return true;
        } else {
            return false;
        }
    }

    static void recursive_clone(Node **c_n, Node *n) {
        if (n == nullptr) {
            return;
        } else {
            *c_n = new Node{};
            *c_n->chr = n->chr;
            *c_n->contain = n->contain;
            *c_n->val = n->val;
            recursive_clone(&c_n->left, n->left);
            recursive_clone(&c_n->mid, n->mid);
            recursive_clone(&c_n->right, n->right);
        }
    }

public:

    class Node {
    private:
        friend TernaryTries;
        char chr{};
        bool contain{false};
        Node *left{nullptr}, *mid{nullptr}, *right{nullptr};
        T_Val val{};

        Node() = default;

        ~Node() = default;

    public:

        [[nodiscard]] char getNodeChar() const {
            return chr;
        }

        [[nodiscard]] bool containValue() const {
            return contain;
        }

        /**
         *
         * @return return copy or return pointer
         */
        T_Val getValue() const {
            return val;
        }

        const Node *getLeft() const {
            return left;
        }

        const Node *getMid() const {
            return mid;
        }

        const Node *getRight() const {
            return right;
        }

        Node(const Node &other) = delete;

        Node(Node &&other) = delete;

        Node &operator=(const Node &other) = delete;

        Node &operator=(Node &&other) = delete;


    };

    TernaryTries() = default;

    TernaryTries(const TernaryTries<T_Val> &other) = delete;

    TernaryTries(TernaryTries<T_Val> &&other) noexcept = default;

    TernaryTries &operator=(const TernaryTries &other) = delete;

    TernaryTries &operator=(TernaryTries &&other) noexcept = default;

    TernaryTries clone() {
        TernaryTries<T_Val> cloned{};
        cloned.count = count;
        recursive_clone(&cloned.root, root);
        return std::move(cloned);
    }

    ~TernaryTries() {
        recursive_release(root);
    }

    const Node *getRoot() {
        return root;
    }

    /**
     *
     * @return count of inserted key value pair
     */
    int getCount() {
        return count;
    }

    /**
     *
     * @param key string eky
     * @param ret value
     * @return if key exists
     */
    bool try_get(const string &key, T_Val *ret) {
        if (key.empty()) {
            return false;
        }
        return get_value(root, key, ret);
    }

    bool contain_key(const string &key) {
        return get_value(root, key, nullptr);
    }

    /**
     * insert_node key value pair
     * @param key
     * @param val
     * @param replace whether replace existing value, default true
     */
    void insert(const string &key, const T_Val &val, bool replace = true) {
        root = insert_node(root, key, val, 0, replace);
    }

    /**
     * remove key and return value
     * @param key
     * @param ret the place to save returned value
     * @return
     */
    bool remove(const string &key, T_Val *ret) {
        if (key.empty()) {
            return false;
        }

        if (remove_node(root, nullptr, Direction::NONE, key, 0, ret)) {
            count--;
            return true;
        }
        return false;
    }


    string longestPrefixOf(const string &query) {
        if (query.length() == 0) { return null; }
        int length = 0;
        auto x{root};
        int i = 0;
        while (x != nullptr && i < query.length()) {
            char c{query.at(i)};
            if (c < x.c) { x = x.left; }
            else if (c > x.c) { x = x.right; }
            else {
                i++;
                if (x.contain) { length = i; }
                x = x.mid;
            }
        }
        return query.substr(0, length);
    }
};

#endif //DEV_QHC_CPP_PROJECTS_TERNARYTRIES_HPP
