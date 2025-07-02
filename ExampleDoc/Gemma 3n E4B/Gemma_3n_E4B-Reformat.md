```markdown
# Algorithmic

## 1 Introduction
======================

This document describes the syntax used in the configuration file for xrootd, cmsd, and all other related components**.** Refer to the respective configuration documents on directive details.

All configuration directives start with a prefix identifying the system component to which the directive applies. The prefix is separated by the actual directive keyword by a single period. This allows configuration of all aspects of a system using a single configuration file. The following table lists valid prefixes

|  |  |
| --- | --- |
| Prefix | System Component |
| **acc** | Access control (i.e., authorization) |
| **cms** | Cluster Management Services |
| **dig** | The digFS built-in file system |
| **frm** | File Residency Manager |
| **http** | HTTP protocol plug-in. |
| **ofs** | Open file system coordinating acc, cms & oss components |
| **oss** | **O**pen **s**torage **s**ystem (i.e., file system implementation) |
| **pfc** | Proxy File Cache plug-in |
| **pss** | Proxy Storage Service plug-in |
| **sec** | Security authentication |
| **xrd** | Extended Request Daemon |
| **xrootd** | The xrootd protocol implementation. |
| **all** | Applies the directive to all of the above components. |

Records that do not start with a recognized identifier are ignored. This includes blank record, comment lines (i.e., lines starting with a pound sign, #), and prefixes not immediately followed by a single period. Because each component has a unique prefix, a common configuration file can be used for the whole system. The location of the configuration file is specified on the command line. Refer to the reference manuals for each component on how it locates the configuration files.

This guide documents the basic directive syntax and describes the use of conditional statements and set variables within the configuration file.


## 2 Specifying Conditional Directives
===========================================

The if-fi directives are used to allow you to optionally include directives based on
host and instance name. The syntax for this directive pair is:

```
if [ *hostpat* [. . .] ] [ *conds* ]
  [ *directives when* if *is true* ]

else if [ *hostpat* [. . .] ] [ *conds* ]
  [ *directives when all previous* if’s *are false*
  and this *if* is true]

else
  [ *directives when all previous* if’s *are false* ]
fi
```

*hostpat*: *host*
| *host***+** | *pfx***\*** | *pfx***\****sfx* | *pfx***\****sfx*]

*conds*:  *cond1*
| *cond2* | *cond3*

*cond1*:  **defined**
| **exec**
| **named**

*var*:  **?**varname
| **?~**varname



**Function**

Specify the conditions under which subsequent directives are to be used.

**Parameters**

*hostpat*
The pattern of the host to which subsequent directive applies. All non-applicable hosts ignore all directives until the next else or fi. Host patterns are:

* host  Any host that matches the specified DNS name.
* host+ Any host that has an address that matches any of the addresses assigned to host.
* pfx\* Any host starting with pfx.
* \*sfx Any host ending with sfx.
* pfx\*sfx Any host beginning with pfx and ending with sfx.

*conds*
A list of conditions that must be true for the directive to be applied. Conditions are specified as:

* defined
* exec
* named

*var*
A set variable name or an environmental variable name, varname. “**?**varname” refers to set variable; while “**?~**varname” refers to an environmental variable.

**Defaults**

None. At least one hostpat, defined, exec or the named keyword must be specified.

**Notes**

1) All specified conditions must be true (i.e., hostpat, defined, exec, and named) for the subsequent directives to be used.

2) A double ampersand (**&&**) is used to “and” two or more *named* tests. Be aware that the specified *named* tests must appear in the specific order (i.e. defined before exec and exec before named).

3) A qualified if is an **if** that is preceded by an else on the same line. An unqualified if is an **if** that appears first on a line.

4) Every unqualified if must be followed by a fi. Every fi must be preceded by a qualified or unqualified if.

5) Every else must be preceded by a qualified or unqualified if.

6) Nested unqualified if directives are not allowed.

7) The name anon refers to servers that were not given a name via –n.

8) Some directives allow the “if” to be placed as the rightmost tokens on the associated directive line. For these directives, no “fi” is required as the end of the line determines the if’s scope.

**Examples**

```
if \*slac.stanford.edu named anon
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
  set myVar = $EXPORTPATH
else
  set myVar = /tmp
fi

all.export $myVar
```


## 3 Configuration File Continuation
=========================================

```
continue [[?]{***dirpath*| *filepath*}[*sfx*][*if* *spec*]]
```

**Function**

Specify the file(s) to continue the current configuration file.

**Parameters**

*dirpath*
The path to a directory holding additional configuration files. The directory is scanned for applicable files and each such file, in lexical order, is used to extend the current configuration file. Applicable files are those that

* do not start with a dot (.),
* are not marked as executable files, and
* do not end with the historical suffixes of “.cfsaved”, “.rpmsave”, “.rpmnew”, “.dpkg-old”, “.dpkg-dist”, or “~” (i.e. tilde).

*filepath*
The path to a file that contains additional configuration directives. The file is used as an immediate continuation to the current configuration file. The file is acceptable as long as it does not have the execute bit set.

*sfx*
One or more file suffixes that are allowed to be continuations (files marked as executable are still ignored). This option is only meaningful when *dirpath* is specified. When *filepath* is specified, any specified valid suffix specifications are ignored.

**txt**
The characters that must appear at the end of the filename for the file to be considered applicable.

*spec*
A valid **if** directive clause. The **continue** statement only applies if the clause evaluates to true.

**Notes**

1) A **continue** directive with no arguments simply continues the processing of the current configuration file. An empty argument list may occur due to variable substitution.

2) The **continue** directive is not allowed in a continuation (i.e., a continuation may not continue to another file).

3) The **continue** directive may not appear in an **if-else-fi** clause. Use an inline **if** to control its applicability.

4) Continuations are particularly useful for defining a base configuration that allows site, VM or container specific augmentation.

5) Be very aware that component directives may be cumulative or replaceable. Refer to the specific directive that you wish to alter should it appear in an antecedent configuration file.

**Example**

```
continue /etc/morecfg .cf .cfg .conf if named foobar
ofs.fslib $(myPath)libXrdOfs.so
```


## 4 Using set Variables
=============================

## 4.1 Assigning Variable Values
-----------------------------------

```
set *var*{ **=** {*value* | *varname* } | < *path* }
```

**Function**

Specify the value a set variable must have.

**Parameters**

*varname*
The name of a variable. Variable names may only contain letters and digits and should start with a letter. Case is significant. Variable names may not be longer than 63 characters.

*value*
The value to be assigned to the variable. It must consist of a single non-blank text token no longer than 511 characters.

*path*
The path to the file that contains the value to be assigned to the variable. The file must not be longer than 1023 characters.

**Notes**

1) Unless **$[***var***]** notation is used; use of an environmental variable that has not been set is considered to be a fatal error.

2) Variables may be used in any text line other than a set statement.

