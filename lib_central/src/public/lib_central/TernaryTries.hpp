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

    bool get(Node *n, const string &key, int depth, T_Val *ret) {
        if (n == nullptr) {
            return false;
        }
        char c = key.at(depth);
        if (c < n->chr) { return get(n->left, key, depth); }
        else if (c > n->chr) { return get(n->right, key, depth); }
        else if (depth < key.length() - 1) { return get(n->mid, key, depth + 1); }
        else {
            *ret = n->val;
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

    static Node *insert(Node *n, const string &key, const T_Val &val, int depth, bool replace) {
        char c = key.at(depth);
        if (n == nullptr) {
            n = new Node{};
            n->chr = c;
        }
        if (c < n->c) { n->left = insert(n->left, key, val, depth, replace); }
        else if (c > n->c) { n->right = insert(n->right, key, val, depth, replace); }
        else if (depth < key.length() - 1) { n->mid = insert(n->mid, key, val, depth + 1); }
        else {
            if (!replace && n->contain) {
                return n;
            } else {
                n->val = val;
                n->contain = true;
            }
        }
        return n;
    }

    static bool remove_node(Node *n, Node *p, Direction direct, const string &key, int d, T_Val *ret) {
        return true;
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
    bool getValueOfKey(const string &key, T_Val *ret) {
        if (key.empty()) {
            return false;
        }
        return get(root, key, ret);
    }

    void insert(const string &key, T_Val val, bool replace = true) {
        root = insert(root, key, val, 0, replace);
    }

    bool remove(const string &key, T_Val *ret) {
        if (key.empty()) {
            return false;
        }

        if (remove_node(root, nullptr, Direction::NONE, key, 0, ret)) {
            count--;
            return true;
        } else {
            return false;
        }
    }
};

#endif //DEV_QHC_CPP_PROJECTS_TERNARYTRIES_HPP
