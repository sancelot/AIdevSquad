"""
Tool to generate a high-level summary of a TypeScript/JavaScript.
Optimized for LLM agents to save tokens.
"""

import json
import os
import subprocess
import tempfile
from typing import Dict, List, Any, Optional
from pathlib import Path


class TypeScriptCodeSummary:
    """Class to generate code summaries TypeScript/JavaScript."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.extension = Path(filepath).suffix
        
    def get_code_summary(self) -> Dict[str, Any]:
        """
        Main entry point. Analyzes the file and returns a structured summary.
        
        Returns:
            Dict containing the code summary
        """
        if not os.path.exists(self.filepath):
            return {
                "error": f"File not found: {self.filepath}",
                "filepath": self.filepath
            }
        
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            return {
                "error": "Unable to read file (encoding issue)",
                "filepath": self.filepath
            }
        except Exception as e:
            return {
                "error": f"Error reading file: {str(e)}",
                "filepath": self.filepath
            }
        
        # Essayer d'abord avec le parser Node.js (le plus précis)
        try:
            result = self._parse_with_nodejs(content)
            if result and 'error' not in result:
                return result
        except Exception as e:
            print(f"Node.js parser failed: {e}")
        
        # Fallback sur regex si Node.js n'est pas disponible
        return self._parse_with_regex(content)
    
    def _parse_with_nodejs(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse with @babel/parser via Node.js (most accurate method)."""
        
        # Créer le script Node.js temporaire
        parser_script = """
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const fs = require('fs');

const code = fs.readFileSync(process.argv[2], 'utf-8');

const summary = {
    imports: [],
    exports: [],
    classes: [],
    functions: [],
    interfaces: [],
    types: [],
    enums: [],
    constants: []
};

try {
    const ast = parser.parse(code, {
        sourceType: 'module',
        plugins: [
            'typescript',
            'jsx',
            'decorators-legacy',
            'classProperties',
            'dynamicImport',
            'asyncGenerators',
            'optionalChaining',
            'nullishCoalescingOperator'
        ]
    });
    
    traverse(ast, {
        ImportDeclaration(path) {
            const node = path.node;
            summary.imports.push({
                source: node.source.value,
                specifiers: node.specifiers.map(spec => {
                    if (spec.type === 'ImportDefaultSpecifier') {
                        return { type: 'default', name: spec.local.name };
                    } else if (spec.type === 'ImportNamespaceSpecifier') {
                        return { type: 'namespace', name: spec.local.name };
                    } else {
                        return { 
                            type: 'named', 
                            imported: spec.imported.name,
                            local: spec.local.name 
                        };
                    }
                })
            });
        },
        
        ExportNamedDeclaration(path) {
            const node = path.node;
            if (node.declaration) {
                const decl = node.declaration;
                if (decl.type === 'VariableDeclaration') {
                    decl.declarations.forEach(d => {
                        summary.exports.push({
                            type: 'named',
                            name: d.id.name
                        });
                    });
                } else if (decl.type === 'FunctionDeclaration') {
                    summary.exports.push({
                        type: 'named',
                        name: decl.id.name
                    });
                } else if (decl.type === 'ClassDeclaration') {
                    summary.exports.push({
                        type: 'named',
                        name: decl.id.name
                    });
                }
            }
        },
        
        ExportDefaultDeclaration(path) {
            const node = path.node;
            let name = 'default';
            if (node.declaration.id) {
                name = node.declaration.id.name;
            }
            summary.exports.push({
                type: 'default',
                name: name
            });
        },
        
        ClassDeclaration(path) {
            if (path.parent.type === 'ExportNamedDeclaration' || 
                path.parent.type === 'ExportDefaultDeclaration' ||
                path.parent.type === 'Program') {
                
                const node = path.node;
                const classInfo = {
                    name: node.id ? node.id.name : 'anonymous',
                    superClass: node.superClass ? node.superClass.name : null,
                    methods: [],
                    properties: [],
                    isAbstract: false
                };
                
                if (node.body && node.body.body) {
                    node.body.body.forEach(member => {
                        if (member.type === 'ClassMethod') {
                            classInfo.methods.push({
                                name: member.key.name || member.key.value,
                                kind: member.kind,
                                isStatic: member.static || false,
                                isAsync: member.async || false,
                                params: member.params.map(p => {
                                    if (p.type === 'Identifier') {
                                        return { name: p.name };
                                    } else if (p.type === 'AssignmentPattern') {
                                        return { 
                                            name: p.left.name,
                                            hasDefault: true 
                                        };
                                    }
                                    return { name: 'param' };
                                })
                            });
                        } else if (member.type === 'ClassProperty') {
                            classInfo.properties.push({
                                name: member.key.name || member.key.value,
                                isStatic: member.static || false
                            });
                        }
                    });
                }
                
                summary.classes.push(classInfo);
            }
        },
        
        FunctionDeclaration(path) {
            if (path.parent.type === 'ExportNamedDeclaration' || 
                path.parent.type === 'ExportDefaultDeclaration' ||
                path.parent.type === 'Program') {
                
                const node = path.node;
                summary.functions.push({
                    name: node.id ? node.id.name : 'anonymous',
                    isAsync: node.async || false,
                    isGenerator: node.generator || false,
                    params: node.params.map(p => {
                        if (p.type === 'Identifier') {
                            return { name: p.name };
                        } else if (p.type === 'AssignmentPattern') {
                            return { 
                                name: p.left.name,
                                hasDefault: true 
                            };
                        } else if (p.type === 'RestElement') {
                            return { 
                                name: '...' + (p.argument.name || 'rest'),
                                isRest: true 
                            };
                        }
                        return { name: 'param' };
                    })
                });
            }
        },
        
        TSInterfaceDeclaration(path) {
            const node = path.node;
            const interfaceInfo = {
                name: node.id.name,
                extends: node.extends ? node.extends.map(e => e.expression.name) : [],
                properties: []
            };
            
            if (node.body && node.body.body) {
                node.body.body.forEach(member => {
                    if (member.type === 'TSPropertySignature' && member.key) {
                        interfaceInfo.properties.push({
                            name: member.key.name || member.key.value,
                            optional: member.optional || false
                        });
                    } else if (member.type === 'TSMethodSignature' && member.key) {
                        interfaceInfo.properties.push({
                            name: member.key.name || member.key.value,
                            isMethod: true,
                            optional: member.optional || false
                        });
                    }
                });
            }
            
            summary.interfaces.push(interfaceInfo);
        },
        
        TSTypeAliasDeclaration(path) {
            const node = path.node;
            summary.types.push({
                name: node.id.name
            });
        },
        
        TSEnumDeclaration(path) {
            const node = path.node;
            summary.enums.push({
                name: node.id.name,
                members: node.members.map(m => m.id.name)
            });
        },
        
        VariableDeclaration(path) {
            if (path.parent.type === 'Program') {
                const node = path.node;
                if (node.kind === 'const') {
                    node.declarations.forEach(decl => {
                        if (decl.id.type === 'Identifier') {
                            summary.constants.push({
                                name: decl.id.name
                            });
                        }
                    });
                }
            }
        }
    });
    
    console.log(JSON.stringify(summary, null, 2));
} catch (error) {
    console.error(JSON.stringify({ error: error.message }));
    process.exit(1);
}
"""
        
        # Sauvegarder le code source dans un fichier temporaire
        with tempfile.NamedTemporaryFile(mode='w', suffix=self.extension, delete=False) as src_file:
            src_file.write(content)
            src_filepath = src_file.name
        
        # Sauvegarder le script parser
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as parser_file:
            parser_file.write(parser_script)
            parser_filepath = parser_file.name
        
        try:
            # Exécuter le parser Node.js
            result = subprocess.run(
                ['node', parser_filepath, src_filepath],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                parsed_data = json.loads(result.stdout)
                
                # Ajouter les métadonnées
                parsed_data['filepath'] = self.filepath
                parsed_data['language'] = 'typescript' if self.extension in ['.ts', '.tsx'] else 'javascript'
                parsed_data['parser'] = 'babel'
                
                # Générer le résumé textuel
                parsed_data['summary'] = self._generate_summary_text(parsed_data)
                
                # Statistiques
                parsed_data['stats'] = {
                    'num_imports': len(parsed_data.get('imports', [])),
                    'num_classes': len(parsed_data.get('classes', [])),
                    'num_functions': len(parsed_data.get('functions', [])),
                    'num_interfaces': len(parsed_data.get('interfaces', [])),
                    'num_types': len(parsed_data.get('types', [])),
                    'num_enums': len(parsed_data.get('enums', []))
                }
                
                # Estimation des tokens
                parsed_data['token_estimate'] = len(json.dumps(parsed_data)) // 4
                
                return parsed_data
            else:
                return None
                
        finally:
            # Nettoyage
            try:
                os.unlink(src_filepath)
                os.unlink(parser_filepath)
            except:
                pass
    
    def _parse_with_regex(self, content: str) -> Dict[str, Any]:
        """Parse with regex (fallback if Node.js is not available)."""
        lines = content.split('\n')
        
        imports = []
        exports = []
        classes = []
        functions = []
        interfaces = []
        types = []
        enums = []
        constants = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Imports
            if line_stripped.startswith('import '):
                imports.append(self._parse_import_line(line_stripped))
            
            # Exports
            if 'export ' in line_stripped:
                exports.append(self._parse_export_line(line_stripped))
            
            # Classes
            if 'class ' in line_stripped:
                class_info = self._parse_class_regex(lines, i)
                if class_info:
                    classes.append(class_info)
            
            # Functions
            if line_stripped.startswith('function ') or \
               line_stripped.startswith('async function ') or \
               line_stripped.startswith('export function ') or \
               line_stripped.startswith('export async function '):
                func_info = self._parse_function_regex(line_stripped)
                if func_info:
                    functions.append(func_info)
            
            # Arrow functions assigned to const
            if line_stripped.startswith('const ') and '=>' in line_stripped:
                func_info = self._parse_arrow_function_regex(line_stripped)
                if func_info:
                    functions.append(func_info)
            
            # Interfaces
            if line_stripped.startswith('interface ') or \
               line_stripped.startswith('export interface '):
                interface_info = self._parse_interface_regex(lines, i)
                if interface_info:
                    interfaces.append(interface_info)
            
            # Type aliases
            if line_stripped.startswith('type ') or \
               line_stripped.startswith('export type '):
                type_info = self._parse_type_regex(line_stripped)
                if type_info:
                    types.append(type_info)
            
            # Enums
            if line_stripped.startswith('enum ') or \
               line_stripped.startswith('export enum '):
                enum_info = self._parse_enum_regex(lines, i)
                if enum_info:
                    enums.append(enum_info)
            
            # Constants
            if line_stripped.startswith('const '):
                const_info = self._parse_constant_regex(line_stripped)
                if const_info:
                    constants.append(const_info)
        
        result = {
            'filepath': self.filepath,
            'language': 'typescript' if self.extension in ['.ts', '.tsx'] else 'javascript',
            'parser': 'regex',
            'imports': imports,
            'exports': exports,
            'classes': classes,
            'functions': functions,
            'interfaces': interfaces,
            'types': types,
            'enums': enums,
            'constants': constants[:10],  # Limiter pour économiser les tokens
            'stats': {
                'num_imports': len(imports),
                'num_classes': len(classes),
                'num_functions': len(functions),
                'num_interfaces': len(interfaces),
                'num_types': len(types),
                'num_enums': len(enums)
            }
        }
        
        result['summary'] = self._generate_summary_text(result)
        result['token_estimate'] = len(json.dumps(result)) // 4
        result['note'] = 'Parsed with regex (install Node.js and @babel/parser for better accuracy)'
        
        return result
    
    def _parse_import_line(self, line: str) -> Dict:
        """Parse an import line."""
        # Exemples:
        # import React from 'react';
        # import { useState, useEffect } from 'react';
        # import * as Utils from './utils';
        
        import re
        match = re.search(r"from\s+['\"]([^'\"]+)['\"]", line)
        source = match.group(1) if match else 'unknown'
        
        return {
            'source': source,
            'raw': line
        }
    
    def _parse_export_line(self, line: str) -> Dict:
        """Parse an export line."""
        if 'export default' in line:
            return {'type': 'default', 'raw': line}
        else:
            return {'type': 'named', 'raw': line}
    
    def _parse_class_regex(self, lines: List[str], start_idx: int) -> Optional[Dict]:
        """Parse a class with regex."""
        line = lines[start_idx].strip()
        
        # Extraire le nom de la classe
        import re
        match = re.search(r'class\s+(\w+)', line)
        if not match:
            return None
        
        class_name = match.group(1)
        
        # Chercher extends
        extends_match = re.search(r'extends\s+(\w+)', line)
        super_class = extends_match.group(1) if extends_match else None
        
        # Parser les méthodes (simpliste)
        methods = []
        properties = []
        
        brace_count = 0
        in_class = False
        
        for i in range(start_idx, min(start_idx + 200, len(lines))):
            current_line = lines[i].strip()
            
            if '{' in current_line:
                brace_count += current_line.count('{')
                in_class = True
            if '}' in current_line:
                brace_count -= current_line.count('}')
            
            if in_class and brace_count > 0:
                # Détecter les méthodes
                method_match = re.match(r'(async\s+)?(\w+)\s*\(', current_line)
                if method_match and not current_line.startswith('//'):
                    methods.append({
                        'name': method_match.group(2),
                        'isAsync': bool(method_match.group(1))
                    })
            
            if in_class and brace_count == 0:
                break
        
        return {
            'name': class_name,
            'superClass': super_class,
            'methods': methods[:10],  # Limiter
            'properties': properties
        }
    
    def _parse_function_regex(self, line: str) -> Optional[Dict]:
        """Parse a function declaration."""
        import re
        
        # function myFunc(arg1, arg2) ou async function myFunc(...)
        match = re.search(r'(async\s+)?function\s+(\w+)\s*\(([^)]*)\)', line)
        if not match:
            return None
        
        is_async = bool(match.group(1))
        name = match.group(2)
        params_str = match.group(3)
        
        params = []
        if params_str:
            for param in params_str.split(','):
                param = param.strip()
                if param:
                    # Extraire juste le nom du paramètre (avant : ou =)
                    param_name = re.split(r'[:\=]', param)[0].strip()
                    params.append({'name': param_name})
        
        return {
            'name': name,
            'isAsync': is_async,
            'params': params
        }
    
    def _parse_arrow_function_regex(self, line: str) -> Optional[Dict]:
        """Parse an arrow function."""
        import re
        
        # const myFunc = (arg1, arg2) => ...
        # const myFunc = async (arg1, arg2) => ...
        match = re.search(r'const\s+(\w+)\s*=\s*(async\s+)?\(([^)]*)\)\s*=>', line)
        if not match:
            # const myFunc = arg => ...
            match = re.search(r'const\s+(\w+)\s*=\s*(async\s+)?(\w+)\s*=>', line)
            if match:
                return {
                    'name': match.group(1),
                    'isAsync': bool(match.group(2)),
                    'params': [{'name': match.group(3)}],
                    'isArrow': True
                }
            return None
        
        name = match.group(1)
        is_async = bool(match.group(2))
        params_str = match.group(3)
        
        params = []
        if params_str:
            for param in params_str.split(','):
                param = param.strip()
                if param:
                    param_name = re.split(r'[:\=]', param)[0].strip()
                    params.append({'name': param_name})
        
        return {
            'name': name,
            'isAsync': is_async,
            'params': params,
            'isArrow': True
        }
    
    def _parse_interface_regex(self, lines: List[str], start_idx: int) -> Optional[Dict]:
        """Parse an interface."""
        import re
        
        line = lines[start_idx].strip()
        match = re.search(r'interface\s+(\w+)', line)
        if not match:
            return None
        
        name = match.group(1)
        
        # Chercher extends
        extends_match = re.search(r'extends\s+([\w,\s]+)', line)
        extends_list = []
        if extends_match:
            extends_list = [e.strip() for e in extends_match.group(1).split(',')]
        
        # Parser les propriétés
        properties = []
        brace_count = 0
        in_interface = False
        
        for i in range(start_idx, min(start_idx + 100, len(lines))):
            current_line = lines[i].strip()
            
            if '{' in current_line:
                brace_count += current_line.count('{')
                in_interface = True
            if '}' in current_line:
                brace_count -= current_line.count('}')
            
            if in_interface and brace_count > 0:
                # Propriété: name: type ou name?: type
                prop_match = re.match(r'(\w+)(\?)?:', current_line)
                if prop_match:
                    properties.append({
                        'name': prop_match.group(1),
                        'optional': bool(prop_match.group(2))
                    })
            
            if in_interface and brace_count == 0:
                break
        
        return {
            'name': name,
            'extends': extends_list,
            'properties': properties[:15]  # Limiter
        }
    
    def _parse_type_regex(self, line: str) -> Optional[Dict]:
        """Parse a type alias."""
        import re
        
        match = re.search(r'type\s+(\w+)', line)
        if not match:
            return None
        
        return {
            'name': match.group(1)
        }
    
    def _parse_enum_regex(self, lines: List[str], start_idx: int) -> Optional[Dict]:
        """Parse an enum."""
        import re
        
        line = lines[start_idx].strip()
        match = re.search(r'enum\s+(\w+)', line)
        if not match:
            return None
        
        name = match.group(1)
        members = []
        
        brace_count = 0
        in_enum = False
        
        for i in range(start_idx, min(start_idx + 50, len(lines))):
            current_line = lines[i].strip()
            
            if '{' in current_line:
                brace_count += current_line.count('{')
                in_enum = True
            if '}' in current_line:
                brace_count -= current_line.count('}')
            
            if in_enum and brace_count > 0:
                # Membre: NAME = value ou NAME,
                member_match = re.match(r'(\w+)', current_line)
                if member_match and member_match.group(1) != 'enum':
                    members.append(member_match.group(1))
            
            if in_enum and brace_count == 0:
                break
        
        return {
            'name': name,
            'members': members
        }
    
    def _parse_constant_regex(self, line: str) -> Optional[Dict]:
        """Parse a constant."""
        import re
        
        # const MY_CONST = ...
        # Seulement les constantes en UPPER_CASE
        match = re.match(r'const\s+([A-Z][A-Z0-9_]*)\s*[=:]', line)
        if not match:
            return None
        
        return {
            'name': match.group(1)
        }
    
    def _generate_summary_text(self, parsed_data: Dict) -> str:
        """Génère un résumé textuel."""
        parts = []
        
        if parsed_data.get('classes'):
            class_names = [c['name'] for c in parsed_data['classes']]
            parts.append(f"{len(class_names)} class(es): {', '.join(class_names)}")
        
        if parsed_data.get('interfaces'):
            interface_names = [i['name'] for i in parsed_data['interfaces']]
            if len(interface_names) > 5:
                parts.append(f"{len(interface_names)} interface(s) including: {', '.join(interface_names[:5])}...")
            else:
                parts.append(f"{len(interface_names)} interface(s): {', '.join(interface_names)}")
        
        if parsed_data.get('functions'):
            func_names = [f['name'] for f in parsed_data['functions']]
            if len(func_names) > 5:
                parts.append(f"{len(func_names)} function(s) including: {', '.join(func_names[:5])}...")
            else:
                parts.append(f"{len(func_names)} function(s): {', '.join(func_names)}")
        
        if parsed_data.get('types'):
            parts.append(f"{len(parsed_data['types'])} type(s)")
        
        if parsed_data.get('enums'):
            parts.append(f"{len(parsed_data['enums'])} enum(s)")
        
        if not parts:
            return "Empty file or no structures detected"
        
        return " | ".join(parts)


def get_typescript_summary(filename: str) -> Dict[str, Any]:
    """
    Fonction utilitaire principale pour obtenir un résumé de code TypeScript/JavaScript.
    
    Args:
        filename: Chemin vers le fichier à analyser
        
    Returns:
        Dictionnaire contenant le résumé structuré du code
    """
    summarizer = TypeScriptCodeSummary(filename)
    return summarizer.get_code_summary()


def get_typescript_summary_formatted(filename: str) -> str:
    """
    Returns a formatted summary in readable text for an LLM agent.
    
    Args:
        filename: Chemin vers le fichier à analyser
        
    Returns:
        String formaté avec le résumé du code
    """
    summary = get_typescript_summary(filename)
    
    if "error" in summary:
        return f"❌ Error: {summary['error']}"
    
    output = []
    output.append(f"📄 File: {summary['filepath']}")
    output.append(f"🔤 Language: {summary['language']}")
    output.append(f"🔧 Parser: {summary.get('parser', 'unknown')}")
    output.append(f"📊 Summary: {summary['summary']}")
    output.append("")
    
    if summary.get('imports'):
        output.append("📦 IMPORTS:")
        for imp in summary['imports'][:10]:
            if isinstance(imp, dict):
                if 'source' in imp:
                    if 'specifiers' in imp:
                        for spec in imp['specifiers'][:3]:
                            output.append(f"  - {spec['type']}: {spec.get('imported', spec.get('name', '?'))} from '{imp['source']}'")
                    else:
                        output.append(f"  - from '{imp['source']}'")
            else:
                output.append(f"  - {imp}")
        if len(summary['imports']) > 10:
            output.append(f"  ... and {len(summary['imports']) - 10} more")
        output.append("")
    
    if summary.get('interfaces'):
        output.append("🔷 INTERFACES:")
        for iface in summary['interfaces']:
            extends = f" extends {', '.join(iface['extends'])}" if iface.get('extends') else ""
            output.append(f"  interface {iface['name']}{extends}:")
            for prop in iface.get('properties', [])[:10]:
                optional = '?' if prop.get('optional') else ''
                method = ' (method)' if prop.get('isMethod') else ''
                output.append(f"    - {prop['name']}{optional}{method}")
            if len(iface.get('properties', [])) > 10:
                output.append(f"    ... and {len(iface['properties']) - 10} more properties")
        output.append("")
    
    if summary.get('types'):
        output.append("📐 TYPE ALIASES:")
        for type_alias in summary['types']:
            output.append(f"  - type {type_alias['name']}")
        output.append("")
    
    if summary.get('enums'):
        output.append("🔢 ENUMS:")
        for enum in summary['enums']:
            output.append(f"  enum {enum['name']}:")
            for member in enum.get('members', [])[:10]:
                output.append(f"    - {member}")
            if len(enum.get('members', [])) > 10:
                output.append(f"    ... and {len(enum['members']) - 10} more members")
        output.append("")
    
    if summary.get('classes'):
        output.append("🏛️  CLASSES:")
        for cls in summary['classes']:
            extends = f" extends {cls['superClass']}" if cls.get('superClass') else ""
            output.append(f"  class {cls['name']}{extends}:")
            for method in cls.get('methods', [])[:10]:
                async_prefix = "async " if method.get('isAsync') else ""
                static_prefix = "static " if method.get('isStatic') else ""
                params = ', '.join([p['name'] for p in method.get('params', [])])
                output.append(f"    - {static_prefix}{async_prefix}{method['name']}({params})")
            if len(cls.get('methods', [])) > 10:
                output.append(f"    ... and {len(cls['methods']) - 10} more methods")
        output.append("")
    
    if summary.get('functions'):
        output.append("⚙️  FUNCTIONS:")
        for func in summary['functions']:
            async_prefix = "async " if func.get('isAsync') else ""
            arrow = " (arrow)" if func.get('isArrow') else ""
            params = ', '.join([p['name'] for p in func.get('params', [])])
            output.append(f"  - {async_prefix}{func['name']}({params}){arrow}")
        output.append("")
    
    if summary.get('constants'):
        output.append("🌐 CONSTANTS:")
        for const in summary['constants'][:10]:
            output.append(f"  - {const['name']}")
        if len(summary['constants']) > 10:
            output.append(f"  ... and {len(summary['constants']) - 10} more")
        output.append("")
    
    if 'token_estimate' in summary:
        output.append(f"💾 Estimated tokens in summary: ~{summary['token_estimate']}")
    
    if 'note' in summary:
        output.append(f"\n⚠️  Note: {summary['note']}")
    
    return '\n'.join(output)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python get_typescript_summary.py <filepath>")
        print("\nExemple:")
        print("  python get_typescript_summary.py App.tsx")
        print("  python get_typescript_summary.py --json server.ts")
        sys.exit(1)
    
    show_json = '--json' in sys.argv
    filepath = sys.argv[-1]
    
    if show_json:
        summary = get_typescript_summary(filepath)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(get_typescript_summary_formatted(filepath))