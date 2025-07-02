# Table of Contents

1. [Introduction](#1-introduction)
2. [Specifying Conditional Directives](#2-specifying-conditional-directives)
3. [Configuration File Continuation](#3-configuration-file-continuation)
4. [Using set Variables](#4-using-set-variables)
   - [Assigning Variable Values](#41-assigning-variable-values)
   - [Substituting Variables](#42-substituting-variables)
   - [Specifying set Options](#43-specifying-set-options)
   - [Assigning Environmental Variable Values](#44-assigning-environmental-variable-values)
5. [Document Change History](#5-document-change-history)

---

## 1. Introduction

This document describes the syntax used in the configuration file for **xrootd**, **cmsd**, and all other related components. Refer to the respective configuration documents on directive details.

All configuration directives start with a prefix identifying the system component to which the directive applies. The prefix is separated from the actual directive keyword by a single period. This allows configuration of all aspects of a system using a single configuration file. The following table lists valid prefixes:

| Prefix   | System Component                                                 |
| -------- | ---------------------------------------------------------------- |
| `acc`    | Access control (authorization)                                   |
| `cms`    | Cluster Management Services                                      |
| `dig`    | The digFS built-in file system                                   |
| `frm`    | File Residency Manager                                           |
| `http`   | HTTP protocol plug-in                                            |
| `ofs`    | Open file system coordinating `acc`, `cms`, and `oss` components |
| `oss`    | Open storage system (file system implementation)                 |
| `pfc`    | Proxy File Cache plug-in                                         |
| `pss`    | Proxy Storage Service plug-in                                    |
| `sec`    | Security authentication                                          |
| `xrd`    | Extended Request Daemon                                          |
| `xrootd` | xrootd protocol implementation                                   |
| `all`    | Applies to all of the above components                           |

Records that do not start with a recognized identifier are ignored. This includes blank lines, comment lines (lines starting with `#`), and prefixes not immediately followed by a period. Because each component has a unique prefix, a single configuration file can be used for the entire system. The location of the configuration file is specified on the command line. See the reference manual for each component.

This guide documents directive syntax and the use of conditional statements and `set` variables.

---

## 2. Specifying Conditional Directives

The `if` ... `fi` directives optionally include configuration lines based on host and instance name:

```text
if [hostpat [...]] [conds]
  [directives if true]
else if [hostpat [...]] [conds]
  [directives if previous ifs are false and this one is true]
...
else
  [directives if all ifs are false]
fi
```

**hostpat options:**

- `host` — matches DNS name
- `host+` — matches host address
- `pfx*` — host starts with prefix
- `*sfx` — host ends with suffix
- `pfx*sfx` — host starts and ends with specified parts

**conds options:**

- `defined var [...]` — variable(s) must be defined
- `exec pgm [...]` — matches executable prefix-name
- `named name [...]` — matches instance name from `-n`

**var forms:** `?varname`, `?~varname`

### Notes

1. All conditions must be true.
2. Use `&&` to combine tests in order: `defined` → `exec` → `named`.
3. Qualified `if` follows an `else`; unqualified starts a line.
4. Each `if` must be closed by a `fi`.
5. Each `else` must be preceded by an `if`.
6. No nested unqualified `if` directives.
7. `anon` refers to unnamed servers.
8. Inline `if` does not require `fi`.

### Example

```text
if *slac.stanford.edu named anon
  xrd.port 9999
fi

if named public
  xrd.port 8888
else
  xrd.port 9999
fi

if exec cmsd && named public
  xrd.port 2131
else
  xrd.port 1094
fi

if defined ?~EXPORTPATH
  set exportpath = $EXPORTPATH
else
  set exportpath = /tmp
fi
all.export $exportpath
```

---

## 3. Configuration File Continuation

```text
continue [[?]{dirpath | filepath}] [sfx] [if spec]
```

### Parameters

- `dirpath`: Directory path scanned for files in lexical order.

  - Exclude files starting with `.`
  - Exclude executables
  - Exclude files with suffixes: `.cfsaved`, `.rpmsave`, `.rpmnew`, `.dpkg-old`, `.dpkg-dist`, `~`

- `filepath`: Path to a non-executable file to include directly

- `sfx`: Allowed file suffixes (used only with `dirpath`)

- `spec`: A conditional `if` statement

### Notes

1. `continue` without arguments continues the same file.
2. `continue` cannot be nested.
3. Cannot appear inside `if-else-fi` blocks.
4. Useful for modular config design.
5. Check whether directives are cumulative or replace previous ones.

### Example

```text
continue /etc/morecfg .cf .cfg .conf if named foobar
```

---

## 4. Using set Variables

### 4.1 Assigning Variable Values

```text
set var = { value | varname }  
set var < path

varname forms: $envvar, $(envvar), ${envvar}, $[envvar]
```

| Specification | Defined envvar | Undefined envvar |
| ------------- | -------------- | ---------------- |
| `$envvar`     | Substituted    | Fatal error      |
| `$(envvar)`   | Substituted    | Fatal error      |
| `${envvar}`   | Substituted    | Fatal error      |
| `$[envvar]`   | Substituted    | Null string      |

### Notes

- Unless `$[envvar]` is used, undefined variables cause fatal error.

**Example**

```text
set myVar = myToken
set yourVar = $EnvVar
```

### 4.2 Substituting Variables

Variables can be used within text lines:

```text
$var, ${var}, $(var), $[var]
```

| Specification | Defined var | Undefined var |
| ------------- | ----------- | ------------- |
| `$var`        | Substituted | Fatal error   |
| `$(var)`      | Substituted | Fatal error   |
| `${var}`      | Substituted | Fatal error   |
| `$[var]`      | Substituted | Null string   |

### Notes

1. Undefined variables error unless `$[var]` is used.
2. Cannot be used as the first token.
3. Substitution is single-pass.

**Example**

```text
set myHost = io.slac.stanford.edu
set myPath = /foo/fum/fi/
all.role manager if $myHost
ofs.fslib $(myPath)libXrdOfs.so
```

### 4.3 Specifying set Options

```text
set -q | -v | -V
```

- `-q`: Quiet mode — no display
- `-v`: Verbose — substituted lines only *(default)*
- `-V`: Very verbose — shows substitutions and lines

### Notes

1. Each component applying the config may echo values depending on verbosity.
2. `-V` causes value assignments to be logged too.

**Example**

```text
set -V
```

### 4.4 Assigning Environmental Variable Values

```text
setenv envvar = { value | varname }  
setenv envvar < path

varname forms: $var, $(var), ${var}, $[var]
```

| Specification | Defined var | Undefined var |
| ------------- | ----------- | ------------- |
| `$var`        | Substituted | Fatal error   |
| `$(var)`      | Substituted | Fatal error   |
| `${var}`      | Substituted | Fatal error   |
| `$[var]`      | Substituted | Null string   |

### Notes

- Variable names must not start with `XRD`
- Must begin with a letter and be ≤ 63 characters

**Example**

```text
setenv EnvVar = myToken
set myPath = /foo/fum/fi/
setenv EnvPath = $[myPath]
```

---

## 5. Document Change History

| Date           | Change Description                                          |
| -------------- | ----------------------------------------------------------- |
| 29 March 2007  | Manual introduced                                           |
| 8 January 2008 | Deprecated `odc` and `olb` components                       |
| 23 June 2009   | Documented `if/else if/else/fi` construct                   |
| 13 Nov 2010    | Added `setenv`, `$[]` syntax; removed `odc`, `olb`          |
| 4 May 2014     | Added `defined` test; `http` component                      |
| 14 Aug 2015    | Added `dig`, `frm`, `pfc`, `pss` components; explained `&&` |
| 27 July 2018   | Documented `continue` directive                             |
| 6 April 2022   | `set`/`setenv` value from file support added                |

