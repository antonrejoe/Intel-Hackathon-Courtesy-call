import Image from "next/image";
import { getCookie } from 'cookies-next';
import Link from 'next/link'
import Layout from "../../components/layout";

async function getServerSideProps(context) {
  const { req, res } = context;
  const username = getCookie('username', { req, res }) || false;
  return { props: { username } };
}

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
     <Layout pageTitle="Home">
        {getServerSideProps.props ?
        <>
            <h2>Hi {getServerSideProps.props}</h2>
            <Link href="/profile">Profile</Link><br/>
            <Link href="/api/logout">Logout</Link>
        </>: 
        <>
            <h2>Log in</h2>
            <Link href="/login">Login</Link><br/>
            <Link href="/signup">Signup</Link>
        </>
        }
        </Layout> 
    </main>
  );
}
