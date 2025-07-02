# 1. Introduction

This document describes the syntax used in the configuration file for **xrootd**, **cmsd**, and all other related components. Refer to the respective configuration documents on directive details.

All configuration directives start with a prefix identifying the system component to which the directive applies. The prefix is separated by the actual directive keyword by a single period. This allows configuration of all aspects of a system using a single configuration file. The following table lists valid prefixes:

| Prefix System Component |                                                                    |
| ----------------------- | ------------------------------------------------------------------ |
| `acc`                   | **Access control** (i.e., authorization)                           |
| `cms`                   | **Cluster Management Services**                                    |
| `dig`                   | The **digFS** built-in file system                                 |
| `frm`                   | **File Residency Manager**                                         |
| `http`                  | **HTTP protocol plug-in**                                          |
| `ofs`                   | **Open file system** coordinating `acc`, `cms`, & `oss` components |
| `oss`                   | **Open storage system** (i.e., file system implementation)         |
| `pfc`                   | **Proxy File Cache** plug-in                                       |
| `pss`                   | **Proxy Storage Service** plug-in                                  |
| `sec`                   | **Security** authentication                                        |
| `xrd`                   | **Extended Request Daemon**                                        |
| `xrootd`                | The **xrootd** protocol implementation                             |
| `all`                   | Applies the directive to all of the above components               |

Records that do not start with a recognized identifier are ignored. This includes blank records, comment lines (i.e., lines starting with a pound sign, `#`), and prefixes not immediately followed by a single period. Because each component has a unique prefix, a common configuration file can be used for the whole system. The location of the configuration file is specified on the command line. Refer to the reference manuals for each component on how it locates the configuration files.

This guide documents the basic directive syntax and describes the use of conditional statements and set variables within the configuration file.

---

# 2. Specifying Conditional Directives

The `if`-`fi` directives are used to allow you to optionally include directives based on host and instance name. The syntax for this directive pair is:

```
if [ hostpat [...] ] [ conds ]
  [ directives when if is true ]
  [ else if [ hostpat [...] ] [ conds ]
    [ directives when all previous if's are false and this if is true ]
  ]
  ...
  [ else
    [ directives when all previous if's are false ]
  ]
fi

```

### Patterns

- `hostpat`: `host` | `host+` | `pfx*` | `*sfx` | `pfx*sfx`
- `conds`: `cond1` | `cond2` | `cond3`
- `cond1`: `defined var [...] [&& {cond1 | cond2 | cond3}]`
- `cond2`: `exec pgm [...] [&& cond3]`
- `cond3`: `named name [...]`
- `var`: `?varname` | `?~varname`

### Function

Specify the conditions under which subsequent directives are to be used.

### Parameters

- **hostpat**: The pattern of the host to which subsequent directive applies. Non-applicable hosts ignore all directives until the next `else` or `fi`.
  - `host`: Any host that matches the specified DNS name.
  - `host+`: Any host that has an address that matches any of the addresses assigned to host.
  - `pfx*`: Any host starting with `pfx`.
  - `*sfx`: Any host ending with `sfx`.
  - `pfx*sfx`: Any host beginning with `pfx` and ending with `sfx`.
- **name**: An instance name specified using the `-n` option.
- **pgm**: Prefix-name of the executable (base filename up to first dot).
- **var**: A set variable or an environmental variable.

### Defaults

None. At least one `hostpat`, `defined`, `exec`, or `named` keyword must be specified.

### Notes

1. All conditions must be true for directives to apply.
2. Use `&&` to combine conditions; order must be `defined` → `exec` → `named`.
3. An unqualified `if` appears at the start of a line; a qualified `if` follows `else`.
4. Every `if` must be followed by a `fi`.
5. Every `else` must be preceded by an `if`.
6. Nested unqualified `if` directives are not allowed.
7. The name `anon` refers to unnamed servers.
8. Some directives support inline `if` without requiring a closing `fi`.

### Examples

```
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
else if exec cmsd && named private
  xrd.port 3121
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

# 3. Configuration File Continuation

### Syntax

```
continue [[?]{dirpath | filepath}] [sfx] [if spec]

```

### Parameters

- **dirpath**: Directory path scanned for files that:
  - Do not start with a dot (`.`)
  - Are not executable
  - Do not end with `.cfsaved`, `.rpmsave`, `.rpmnew`, `.dpkg-old`, `.dpkg-dist`, or `~`
- **filepath**: A non-executable file containing directives to continue the current file.
- **sfx**: Suffix(es) used to filter applicable files in `dirpath`.
- **spec**: An inline `if` condition that must be true for the continuation to apply.

### Notes

1. A `continue` directive with no arguments continues the current file.
2. Continuations cannot themselves use `continue`.
3. Continuations may not appear inside `if-else-fi` blocks.
4. Useful for layered configurations (e.g., site-specific overrides).
5. Be mindful of whether directives are cumulative or replaceable.

### Example

```
continue /etc/morecfg .cf .cfg .conf if named foobar

```

---

# 4. Using set Variables

## 4.1 Assigning Variable Values

### Syntax

```
set var = { value | varname }
set var < path
```

- `varname`: `$envvar`, `$(envvar)`, `${envvar}`, `$[envvar]`

### Function

Specify the value a set variable must have.

### Parameters

- **var**: The name of a variable. Must start with a letter, only contain letters/digits, case-sensitive, and be no longer than 63 characters.

- **value**: A single non-blank text token no longer than 511 characters.

- **path**: Path to a file containing the value. File must not exceed 1023 characters.

- **varname**: Environmental variable name (511 character max). If undefined:

| Specification | Defined `envvar`       | Undefined `envvar`      |
| ------------- | ---------------------- | ----------------------- |
| `$envvar`     | Definition substituted | Fatal error             |
| `$(envvar)`   | Definition substituted | Fatal error             |
| `${envvar}`   | Definition substituted | Fatal error             |
| `$[envvar]`   | Definition substituted | Null string substituted |

### Notes

1. Unless `$[envvar]` is used, referencing an undefined environment variable results in a fatal error.

## 4.2 Substituting Variables

### Syntax

```
$varname
```

Substitute a set variable. Used in directive values.

### Function

Allows for dynamic configuration by referencing previously defined variables.

### Parameters

- **varname**: The name of a previously defined variable.

### Notes

1. If a variable is not defined when referenced, a fatal error occurs.
2. Variables can be nested.

## 4.3 Specifying set Options

### Syntax

```
setopt option=value
```

### Function

Define interpreter options that influence how configuration directives behave.

### Options

- `trace`: Enable configuration trace output.
- `warn`: Show warnings on undefined variables.

## 4.4 Assigning Environmental Variable Values

### Syntax

```
setenv var = value
```

### Function

Set an environment variable from within the configuration.

### Parameters

- **var**: Name of the environment variable.
- **value**: Value to assign.

---

# 5. Document Change History

| Date       | Change Description                           |
| ---------- | -------------------------------------------- |
| YYYY-MM-DD | Initial creation of document                 |
| YYYY-MM-DD | Added conditional directive section          |
| YYYY-MM-DD | Updated variable substitution behavior notes |

