#pragma once

#include <stdio.h>
#include <string.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include <vector>
#include <queue>
#include <limits>
#include <cmath>
#include <algorithm>

#if defined(_MSC_VER) && _MSC_VER == 1500
typedef unsigned char uint8_t;
typedef signed __int32 int32_t;
typedef unsigned __int64 uint64_t;
typedef signed __int64 int64_t;
#else
#include <stdint.h>
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#define off_t int64_t
#define lseek_getsize(fd) _lseeki64(fd, 0, SEEK_END)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "mman.h"
#include <windows.h>
#else
#include <sys/mman.h>
#define lseek_getsize(fd) lseek(fd, 0, SEEK_END)
#endif

#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#if __cplusplus >= 201103L
#include <random>
#endif

#ifdef _MSC_VER
#pragma runtime_checks("s", off)
#endif

#ifndef __ERROR_PRINTER_OVERRIDE__
#define jakubelib_showUpdate(...) { fprintf(stderr, __VA_ARGS__ ); }
#else
#define jakubelib_showUpdate(...) { __ERROR_PRINTER_OVERRIDE__( __VA_ARGS__ ); }
#endif

// Portable alloc definition
#ifdef __GNUC__
# undef alloca
# define alloca(x) __builtin_alloca((x))
#elif defined(__sun) || defined(_AIX)
# include <alloca.h>
#endif

#define JAKUBELIB_V_ARRAY_SIZE 65536

// Popcount for Hamming distance
#ifndef _MSC_VER
#define jakubelib_popcount __builtin_popcountll
#else
#define jakubelib_popcount cole_popcount
#endif

#if !defined(__MINGW32__)
#define JAKUBELIB_FTRUNCATE_SIZE(x) static_cast<off_t>(x)
#else
#define JAKUBELIB_FTRUNCATE_SIZE(x) (x)
#endif

namespace Jakube {

inline void set_error_from_errno(char **error, const char* msg) {
  jakubelib_showUpdate("%s: %s (%d)\n", msg, strerror(errno), errno);
  if (error) {
    *error = (char *)malloc(256);
    snprintf(*error, 255, "%s: %s (%d)", msg, strerror(errno), errno);
  }
}

inline void set_error_from_string(char **error, const char* msg) {
  jakubelib_showUpdate("%s\n", msg);
  if (error) {
    *error = (char *)malloc(strlen(msg) + 1);
    strcpy(*error, msg);
  }
}

using std::vector;
using std::pair;
using std::numeric_limits;
using std::make_pair;

inline bool remap_memory_and_truncate(void** _ptr, int _fd, size_t old_size, size_t new_size) {
#ifdef __linux__
  *_ptr = mremap(*_ptr, old_size, new_size, MREMAP_MAYMOVE);
  bool ok = ftruncate(_fd, new_size) != -1;
#else
  munmap(*_ptr, old_size);
  bool ok = ftruncate(_fd, JAKUBELIB_FTRUNCATE_SIZE(new_size)) != -1;
#ifdef MAP_POPULATE
  *_ptr = mmap(*_ptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, _fd, 0);
#else
  *_ptr = mmap(*_ptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
#endif
#endif
  return ok;
}

namespace {
  template<typename Node, typename S>
  inline Node* get_node_ptr(const void* _nodes, const size_t _s, const S i) {
    return (Node*)((uint8_t *)_nodes + (_s * i));
  }
} // namespace

// Base distance struct with common utilities
struct Base {
  template<typename Node, typename T, typename S>
  static inline void preprocess(void* nodes, size_t _s, const S node_count, const int f) {
    // No preprocessing needed for Hamming
  }
  
  template<typename Node, typename T, typename S>
  static inline void postprocess(void* nodes, size_t _s, const S node_count, const int f) {
    // No postprocessing needed for Hamming
  }
  
  template<typename Node>
  static inline void zero_value(Node* dest) {
    // No special initialization needed for Hamming
  }
  
  template<typename Node, typename T>
  static inline void copy_node(Node* dest, const Node* source, const int f) {
    memcpy(dest->v, source->v, f * sizeof(T));
  }

  // Overload: copy directly from a raw values array into the node
  template<typename Node, typename T>
  static inline void copy_node(Node* dest, const T* source_values, const int f) {
    memcpy(dest->v, source_values, f * sizeof(T));
  }
};

// Hamming distance metric (only one we need!)
struct Hamming : Base {
  template<typename S, typename T>
  struct Node {
    S n_descendants;
    S children[2];
    T v[JAKUBELIB_V_ARRAY_SIZE];
  };
  
  static const size_t max_iterations = 20;
  
  template<typename T>
  static inline T pq_distance(T distance, T margin, int child_nr) {
    return distance - (margin != (unsigned int) child_nr);
  }
  
  template<typename T>
  static inline T pq_initial_value() {
    return numeric_limits<T>::max();
  }
  
  // MSVC popcount implementation
  template<typename T>
  static inline int cole_popcount(T v) {
    v = v - ((v >> 1) & (T)~(T)0/3);
    v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3);
    v = (v + (v >> 4)) & (T)~(T)0/255*15;
    return (T)(v * ((T)~(T)0/255)) >> (sizeof(T) - 1) * 8;
  }
  
  template<typename S, typename T>
  static inline T distance(const Node<S,T>* x, const Node<S,T>* y, int f) {
    size_t dist = 0;
    for (int i = 0; i < f; i++) {
      dist += jakubelib_popcount(x->v[i] ^ y->v[i]);
    }
    return dist;
  }
  
  template<typename S, typename T>
  static inline bool margin(const Node<S,T>* n, const T* y, int f) {
    static const size_t n_bits = sizeof(T) * 8;
    T chunk = n->v[0] / n_bits;
    return (y[chunk] & (static_cast<T>(1) << (n_bits - 1 - (n->v[0] % n_bits)))) != 0;
  }
  
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S,T>* n, const T* y, int f, Random& random) {
    return margin(n, y, f);
  }
  
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S,T>* n, const Node<S,T>* y, int f, Random& random) {
    return side(n, y->v, f, random);
  }
  
  template<typename S, typename T, typename Random>
  static inline void create_split(const vector<const Node<S,T>*>& nodes, int f, size_t s, Random& random, Node<S,T>* n) {
    size_t cur_size = 0;
    size_t i = 0;
    int dim = f * 8 * sizeof(T);
    
    // Try random splits first
    for (; i < max_iterations; i++) {
      n->v[0] = random.index(dim);
      cur_size = 0;
      
      for (typename vector<const Node<S,T>*>::const_iterator it = nodes.begin(); it != nodes.end(); ++it) {
        if (margin(n, (*it)->v, f)) {
          cur_size++;
        }
      }
      
      if (cur_size > 0 && cur_size < nodes.size()) {
        break;
      }
    }
    
    // Brute-force if random didn't work
    if (i == max_iterations) {
      int j = 0;
      for (; j < dim; j++) {
        n->v[0] = j;
        cur_size = 0;
        
        for (typename vector<const Node<S,T>*>::const_iterator it = nodes.begin(); it != nodes.end(); ++it) {
          if (margin(n, (*it)->v, f)) {
            cur_size++;
          }
        }
        
        if (cur_size > 0 && cur_size < nodes.size()) {
          break;
        }
      }
    }
  }
  
  template<typename T>
  static inline T normalized_distance(T distance) {
    return distance;
  }
  
  template<typename S, typename T>
  static inline void init_node(Node<S,T>* n, int f) {
    // No initialization needed for Hamming
  }
  
  static const char* name() {
    return "hamming";
  }
};

// Main Jakube Index class
template<typename S, typename T, typename Distance, typename Random>
class JakubeIndex {
public:
  typedef Distance D;
  typedef typename D::template Node<S, T> Node;

protected:
  const int _f;
  size_t _s;
  S _n_items;
  void* _nodes; // Memory for all nodes
  S _n_nodes;
  S _nodes_size;
  vector<S> _roots;
  S _K;
  Random _random;
  bool _verbose;
  bool _built;
  int _fd; // File descriptor for memory-mapped file

public:
  JakubeIndex(int f) : _f(f), _s(offsetof(Node, v) + _f * sizeof(T)), 
                       _n_items(0), _nodes(NULL), _n_nodes(0),
                       _nodes_size(0), _K((S)(((_s - offsetof(Node, children)) / sizeof(S)) - 2)),
                       _random(), _verbose(false), _built(false), _fd(0) {
  }

  ~JakubeIndex() {
    unload();
  }

  int get_f() const {
    return _f;
  }

  bool add_item(S item, const T* w, char** error=NULL) {
    if (_built) {
      set_error_from_string(error, "Index already built, cannot add more items");
      return false;
    }

    _allocate_size(item + 1);
    Node* n = _get(item);
    
    D::zero_value(n);
    n->n_descendants = 1;
    // Copy from raw value buffer into node storage
    D::template copy_node<Node, T>(n, w, _f);
    D::init_node(n, _f);
    
    if (item >= _n_items)
      _n_items = item + 1;
    
    return true;
  }

  bool build(int q, int n_threads=-1, char** error=NULL) {
    if (_built) {
      set_error_from_string(error, "Index already built");
      return false;
    }
    
    if (_n_items == 0) {
      set_error_from_string(error, "No items added");
      return false;
    }

    D::template preprocess<Node, T, S>(_nodes, _s, _n_items, _f);
    
    _n_nodes = _n_items;
    
    for (int i = 0; i < q; i++) {
      vector<S> indices;
      for (S j = 0; j < _n_items; j++) {
        indices.push_back(j);
      }
      
      _roots.push_back(_make_tree(indices, true));
      
      if (_verbose) jakubelib_showUpdate("Built tree %d/%d\n", i+1, q);
    }
    
    if (_verbose) jakubelib_showUpdate("Built %d trees with %d nodes\n", q, _n_nodes);
    
    _built = true;
    D::template postprocess<Node, T, S>(_nodes, _s, _n_items, _f);
    
    return true;
  }

  bool unbuild(char** error=NULL) {
    if (!_built) {
      set_error_from_string(error, "Index not built");
      return false;
    }
    
    _roots.clear();
    _n_nodes = _n_items;
    _built = false;
    
    return true;
  }

  bool save(const char* filename, bool prefault=false, char** error=NULL) {
    if (!_built) {
      set_error_from_string(error, "Index not built");
      return false;
    }

#ifdef _MSC_VER
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, (int)0600);
#else
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, (int)0600);
#endif
    
    if (fd == -1) {
      set_error_from_errno(error, "Unable to open file for writing");
      return false;
    }

    // Write header
    if (write(fd, &_f, sizeof(int)) == -1 ||
        write(fd, &_n_items, sizeof(S)) == -1 ||
        write(fd, &_n_nodes, sizeof(S)) == -1 ||
        write(fd, &_nodes_size, sizeof(S)) == -1 ||
        write(fd, &_K, sizeof(S)) == -1) {
      close(fd);
      set_error_from_errno(error, "Unable to write header");
      return false;
    }

    // Write roots
    S roots_size = (S)_roots.size();
    if (write(fd, &roots_size, sizeof(S)) == -1 ||
        write(fd, _roots.data(), roots_size * sizeof(S)) == -1) {
      close(fd);
      set_error_from_errno(error, "Unable to write roots");
      return false;
    }

    // Write nodes
    if (write(fd, _nodes, _s * _n_nodes) == -1) {
      close(fd);
      set_error_from_errno(error, "Unable to write nodes");
      return false;
    }

    close(fd);
    return true;
  }

  bool load(const char* filename, bool prefault=false, char** error=NULL) {
    unload();

#ifdef _MSC_VER
    _fd = open(filename, O_RDONLY | O_BINARY);
#else
    _fd = open(filename, O_RDONLY);
#endif

    if (_fd == -1) {
      set_error_from_errno(error, "Unable to open file for reading");
      return false;
    }

    // Read header
    int f_file;
    if (read(_fd, &f_file, sizeof(int)) == -1) {
      close(_fd);
      _fd = 0;
      set_error_from_errno(error, "Unable to read f");
      return false;
    }
    
    if (f_file != _f) {
      close(_fd);
      _fd = 0;
      set_error_from_string(error, "Dimension mismatch");
      return false;
    }

    if (read(_fd, &_n_items, sizeof(S)) == -1 ||
        read(_fd, &_n_nodes, sizeof(S)) == -1 ||
        read(_fd, &_nodes_size, sizeof(S)) == -1 ||
        read(_fd, &_K, sizeof(S)) == -1) {
      close(_fd);
      _fd = 0;
      set_error_from_errno(error, "Unable to read header");
      return false;
    }

    // Read roots
    S roots_size;
    if (read(_fd, &roots_size, sizeof(S)) == -1) {
      close(_fd);
      _fd = 0;
      set_error_from_errno(error, "Unable to read roots size");
      return false;
    }
    
    _roots.resize(roots_size);
    if (read(_fd, _roots.data(), roots_size * sizeof(S)) == -1) {
      close(_fd);
      _fd = 0;
      set_error_from_errno(error, "Unable to read roots");
      return false;
    }

    // Memory-map nodes
    size_t nodes_size = _s * _n_nodes;
#ifdef MAP_POPULATE
    _nodes = mmap(0, nodes_size, PROT_READ, MAP_SHARED | MAP_POPULATE, _fd, 0);
#else
    _nodes = mmap(0, nodes_size, PROT_READ, MAP_SHARED, _fd, 0);
#endif

    if (_nodes == MAP_FAILED) {
      close(_fd);
      _fd = 0;
      _nodes = NULL;
      set_error_from_errno(error, "Unable to mmap nodes");
      return false;
    }

    _built = true;
    return true;
  }

  void unload() {
    if (_fd) {
      close(_fd);
      _fd = 0;
    }
    if (_nodes) {
      if (_built) {
        munmap(_nodes, _s * _n_nodes);
      } else {
        free(_nodes);
      }
      _nodes = NULL;
    }
    _roots.clear();
    _n_items = 0;
    _n_nodes = 0;
    _nodes_size = 0;
    _built = false;
  }

  void verbose(bool v) {
    _verbose = v;
  }

  void set_seed(uint64_t seed) {
    _random.set_seed(seed);
  }

  void get_item(S item, T* v) const {
    Node* n = _get(item);
    memcpy(v, n->v, _f * sizeof(T));
  }

  T get_distance(S i, S j) const {
    return D::normalized_distance(D::distance(_get(i), _get(j), _f));
  }

  void get_nns_by_item(S item, size_t n, int search_k, vector<S>* result, vector<T>* distances) const {
    Node* node = _get(item);
    _get_all_nns(node->v, n, search_k, result, distances);
  }

  void get_nns_by_vector(const T* w, size_t n, int search_k, vector<S>* result, vector<T>* distances) const {
    _get_all_nns(w, n, search_k, result, distances);
  }

  S get_n_items() const {
    return _n_items;
  }

  S get_n_trees() const {
    return (S)_roots.size();
  }

  void on_disk_build(const char* file, char** error=NULL) {
    // Simplified: not implementing on-disk build for beginner version
    set_error_from_string(error, "on_disk_build not implemented in simplified version");
  }

protected:
  void _allocate_size(S n) {
    if (n > _nodes_size) {
      const double reallocation_factor = 1.3;
      S new_nodes_size = std::max(n, (S)((_nodes_size + 1) * reallocation_factor));
      void* new_nodes = realloc(_nodes, _s * new_nodes_size);
      
      if (!new_nodes) {
        throw std::bad_alloc();
      }
      
      _nodes = new_nodes;
      _nodes_size = new_nodes_size;
    }
  }

  inline Node* _get(const S i) const {
    return get_node_ptr<Node, S>(_nodes, _s, i);
  }

  S _make_tree(const vector<S>& indices, bool is_root) {
    if (indices.size() == 1 && !is_root)
      return indices[0];

    if (indices.size() <= (size_t)_K && !is_root) {
      _allocate_size(_n_nodes + 1);
      S item = _n_nodes++;
      Node* m = _get(item);
      m->n_descendants = (S)indices.size();
      
      for (size_t i = 0; i < indices.size(); i++) {
        m->children[i] = indices[i];
      }
      return item;
    }

    vector<const Node*> children;
    for (size_t i = 0; i < indices.size(); i++) {
      S j = indices[i];
      Node* n = _get(j);
      if (n)
        children.push_back(n);
    }

    vector<S> children_indices[2];
    Node* m = (Node*)alloca(_s);
    
    for (int attempt = 0; attempt < 3; attempt++) {
      children_indices[0].clear();
      children_indices[1].clear();
      
      D::create_split(children, _f, _s, _random, m);
      
      for (size_t i = 0; i < indices.size(); i++) {
        S j = indices[i];
        Node* n = _get(j);
        if (n) {
          bool side = D::side(m, n, _f, _random);
          children_indices[side].push_back(j);
        }
      }
      
      if (_split_imbalance(children_indices[0], children_indices[1]) < 0.95)
        break;
    }

    // Fallback: randomize if no good split found
    while (_split_imbalance(children_indices[0], children_indices[1]) > 0.99) {
      if (_verbose)
        jakubelib_showUpdate("\tNo hyperplane found (left: %zu, right: %zu)\n",
                            children_indices[0].size(), children_indices[1].size());
      
      children_indices[0].clear();
      children_indices[1].clear();
      
      for (size_t i = 0; i < indices.size(); i++) {
        children_indices[_random.flip()].push_back(indices[i]);
      }
    }

    int flip = (children_indices[0].size() > children_indices[1].size());
    
    m->n_descendants = is_root ? _n_items : (S)indices.size();
    for (int side = 0; side < 2; side++) {
      m->children[side^flip] = _make_tree(children_indices[side^flip], false);
    }

    _allocate_size(_n_nodes + 1);
    S item = _n_nodes++;
    memcpy(_get(item), m, _s);
    
    return item;
  }

  void _get_all_nns(const T* v, size_t n, int search_k, vector<S>* result, vector<T>* distances) const {
    Node* v_node = (Node *)alloca(_s);
    D::zero_value(v_node);
    memcpy(v_node->v, v, sizeof(T) * _f);
    D::init_node(v_node, _f);
    
    std::priority_queue<pair<T, S>> q;
    
    if (search_k == -1) {
      search_k = n * _roots.size();
    }

    for (size_t i = 0; i < _roots.size(); i++) {
      q.push(make_pair(Distance::template pq_initial_value<T>(), _roots[i]));
    }

    vector<S> nns;
    while (nns.size() < (size_t)search_k && !q.empty()) {
      const pair<T, S>& top = q.top();
      T d = top.first;
      S i = top.second;
      Node* nd = _get(i);
      q.pop();
      
      if (nd->n_descendants == 1 && i < _n_items) {
        nns.push_back(i);
      } else if (nd->n_descendants <= _K) {
        const S* dst = nd->children;
        nns.insert(nns.end(), dst, &dst[nd->n_descendants]);
      } else {
        T margin = D::margin(nd, v, _f);
        q.push(make_pair(D::pq_distance(d, margin, 1), static_cast<S>(nd->children[1])));
        q.push(make_pair(D::pq_distance(d, margin, 0), static_cast<S>(nd->children[0])));
      }
    }

    // Calculate distances
    std::sort(nns.begin(), nns.end());
    vector<pair<T, S>> nns_dist;
    S last = -1;
    
    for (size_t i = 0; i < nns.size(); i++) {
      S j = nns[i];
      if (j == last)
        continue;
      last = j;
      
      if (_get(j)->n_descendants == 1)
        nns_dist.push_back(make_pair(D::distance(v_node, _get(j), _f), j));
    }

    size_t m = nns_dist.size();
    size_t p = n < m ? n : m;
    std::partial_sort(nns_dist.begin(), nns_dist.begin() + p, nns_dist.end());
    
    for (size_t i = 0; i < p; i++) {
      if (distances)
        distances->push_back(D::normalized_distance(nns_dist[i].first));
      result->push_back(nns_dist[i].second);
    }
  }

  double _split_imbalance(const vector<S>& left, const vector<S>& right) {
    double ls = (double)left.size();
    double rs = (double)right.size();
    if (ls == 0 || rs == 0) return 1.0;
    return std::max(ls/rs, rs/ls);
  }
};

// Single-threaded build policy (only one we need)
class JakubeIndexSingleThreadedBuildPolicy {
public:
  template<typename S, typename T, typename Distance, typename Random>
  static void build(JakubeIndex<S, T, Distance, Random>* jakube, int q, int n_threads) {
    // Just call build directly - no threading
    char* error = NULL;
    jakube->build(q, -1, &error);
    if (error) {
      free(error);
    }
  }
};

} // namespace Jakube

#ifdef _MSC_VER
#pragma runtime_checks("s", restore)
#endif
