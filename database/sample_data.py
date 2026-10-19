# database/sample_data.py
"""
Generate sample business database for testing
Creates realistic business data: employees, sales, products, customers
"""
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import sqlite3
from pathlib import Path
from utils.logger import logger


def create_sample_database(db_path: str):
    """
    Create sample business database with realistic data

    Tables:
    - employees: Company employees
    - departments: Organizational departments
    - products: Product catalog
    - customers: Customer information
    - sales: Sales transactions
    """

    logger.info("🏗️  Creating sample business database...")

    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ========== DEPARTMENTS TABLE ==========
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS departments (
            department_id INTEGER PRIMARY KEY,
            department_name TEXT NOT NULL,
            budget REAL,
            manager_name TEXT
        )
    ''')

    departments_data = [
        (1, 'Engineering', 500000, 'Alice Johnson'),
        (2, 'Sales', 300000, 'Bob Smith'),
        (3, 'Marketing', 250000, 'Carol White'),
        (4, 'HR', 150000, 'David Brown'),
        (5, 'Finance', 200000, 'Eve Davis')
    ]

    cursor.executemany(
        'INSERT OR IGNORE INTO departments VALUES (?,?,?,?)',
        departments_data
    )

    # ========== EMPLOYEES TABLE ==========
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            employee_id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT,
            department_id INTEGER,
            salary REAL,
            hire_date DATE,
            job_title TEXT,
            FOREIGN KEY (department_id) REFERENCES departments(department_id)
        )
    ''')

    employees_data = [
        (1, 'Alice', 'Johnson', 'alice.j@company.com', 1, 120000, '2019-01-15', 'Engineering Manager'),
        (2, 'Bob', 'Smith', 'bob.s@company.com', 2, 95000, '2020-03-20', 'Sales Manager'),
        (3, 'Charlie', 'Brown', 'charlie.b@company.com', 1, 110000, '2019-06-10', 'Senior Engineer'),
        (4, 'Diana', 'Prince', 'diana.p@company.com', 3, 85000, '2021-02-01', 'Marketing Manager'),
        (5, 'Eve', 'Davis', 'eve.d@company.com', 5, 105000, '2020-11-15', 'Finance Manager'),
        (6, 'Frank', 'Miller', 'frank.m@company.com', 1, 95000, '2020-04-12', 'Software Engineer'),
        (7, 'Grace', 'Lee', 'grace.l@company.com', 2, 78000, '2021-07-22', 'Sales Representative'),
        (8, 'Henry', 'Wilson', 'henry.w@company.com', 1, 92000, '2020-09-18', 'Software Engineer'),
        (9, 'Iris', 'Taylor', 'iris.t@company.com', 3, 72000, '2022-01-10', 'Marketing Specialist'),
        (10, 'Jack', 'Anderson', 'jack.a@company.com', 4, 68000, '2021-05-30', 'HR Specialist')
    ]

    cursor.executemany(
        'INSERT OR IGNORE INTO employees VALUES (?,?,?,?,?,?,?,?)',
        employees_data
    )

    # ========== PRODUCTS TABLE ==========
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT,
            price REAL,
            cost REAL,
            stock_quantity INTEGER
        )
    ''')

    products_data = [
        (1, 'Laptop Pro 15"', 'Electronics', 1299.99, 800.00, 45),
        (2, 'Wireless Mouse', 'Accessories', 29.99, 12.00, 200),
        (3, 'USB-C Hub', 'Accessories', 49.99, 20.00, 150),
        (4, 'Monitor 27"', 'Electronics', 399.99, 250.00, 75),
        (5, 'Mechanical Keyboard', 'Accessories', 129.99, 60.00, 120)
    ]

    cursor.executemany(
        'INSERT OR IGNORE INTO products VALUES (?,?,?,?,?,?)',
        products_data
    )

    # ========== CUSTOMERS TABLE ==========
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            company_name TEXT NOT NULL,
            contact_name TEXT,
            email TEXT,
            country TEXT,
            signup_date DATE
        )
    ''')

    customers_data = [
        (1, 'TechCorp Inc', 'John Smith', 'john@techcorp.com', 'USA', '2023-01-15'),
        (2, 'Digital Solutions', 'Sarah Johnson', 'sarah@digital.com', 'UK', '2023-02-20'),
        (3, 'Innovation Labs', 'Mike Chen', 'mike@innov.com', 'Canada', '2023-03-10'),
        (4, 'Global Systems', 'Emma Wilson', 'emma@global.com', 'Australia', '2023-04-05'),
        (5, 'Smart Business', 'Alex Brown', 'alex@smart.com', 'USA', '2023-05-12')
    ]

    cursor.executemany(
        'INSERT OR IGNORE INTO customers VALUES (?,?,?,?,?,?)',
        customers_data
    )

    # ========== SALES TABLE ==========
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales (
            sale_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id INTEGER,
            employee_id INTEGER,
            quantity INTEGER,
            sale_date DATE,
            total_amount REAL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id),
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
        )
    ''')

    sales_data = [
        (1, 1, 1, 2, 10, '2024-01-15', 12999.90),
        (2, 2, 2, 7, 50, '2024-01-20', 1499.50),
        (3, 3, 5, 2, 25, '2024-02-05', 3249.75),
        (4, 4, 1, 2, 5, '2024-02-10', 6499.95),
        (5, 5, 3, 7, 30, '2024-02-15', 1499.70),
        (6, 1, 4, 2, 8, '2024-03-01', 3199.92),
        (7, 2, 5, 7, 20, '2024-03-15', 2599.80),
        (8, 3, 1, 2, 12, '2024-04-01', 15599.88)
    ]

    cursor.executemany(
        'INSERT OR IGNORE INTO sales VALUES (?,?,?,?,?,?,?)',
        sales_data
    )

    conn.commit()
    conn.close()

    logger.info("✅ Sample database created successfully")
    logger.info(f"  📊 Tables: departments, employees, products, customers, sales")


if __name__ == "__main__":
    create_sample_database("./data/business.db")
    print("\n✅ Database created at: ./data/business.db")